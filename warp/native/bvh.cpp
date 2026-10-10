// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "bvh.h"
#include "cuda_util.h"
#include "error.h"
#include "metal_host.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <vector>

using namespace wp;

namespace wp {


/////////////////////////////////////////////////////////////////////////////////////////////

class TopDownBVHBuilder {
public:
    void build(BVH& bvh, const vec3* lowers, const vec3* uppers, int n, int in_constructor_type, int* groups);
    void rebuild(BVH& bvh, int in_constructor_type);

private:
    void initialize_empty(BVH& bvh);

    bounds3 calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end);
    int build_recursive(
        BVH& bvh,
        const vec3* lowers,
        const vec3* uppers,
        int start,
        int end,
        int depth,
        int parent,
        int assigned_node = -1
    );
    int
    partition_median(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    int
    partition_midpoint(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    float partition_sah_indices(
        const vec3* lowers,
        const vec3* uppers,
        const int* indices,
        int start,
        int end,
        bounds3 range_bounds,
        int& split_axis
    );
    void build_with_groups(BVH& bvh, const vec3* lowers, const vec3* uppers, const int* groups, int n);
    int constructor_type = -1;
};

//////////////////////////////////////////////////////////////////////

void TopDownBVHBuilder::initialize_empty(BVH& bvh)
{
    bvh.max_depth = 0;
    bvh.max_nodes = 0;
    bvh.node_lowers = nullptr;
    bvh.node_uppers = nullptr;
    bvh.node_parents = nullptr;
    bvh.node_counts = nullptr;
    bvh.root = nullptr;
    bvh.primitive_indices = nullptr;
    bvh.num_leaf_nodes = 0;
    bvh.num_items = 0;
    bvh.constructor_type = BVH_CONSTRUCTOR_SAH;
}

static inline void compute_group_ranges(
    const int* groups,
    int n,
    std::vector<int>& unique_groups,
    std::vector<int>& group_starts,
    std::vector<int>& group_ends,
    std::vector<int>& sorted_indices
)
{
    sorted_indices.resize(n);
    for (int i = 0; i < n; ++i)
        sorted_indices[i] = i;
    std::stable_sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) { return groups[a] < groups[b]; });

    unique_groups.clear();
    group_starts.clear();
    group_ends.clear();
    if (n == 0)
        return;

    int current_group = groups[sorted_indices[0]];
    unique_groups.push_back(current_group);
    group_starts.push_back(0);
    for (int i = 1; i < n; ++i) {
        int g = groups[sorted_indices[i]];
        if (g != current_group) {
            group_ends.push_back(i);
            current_group = g;
            unique_groups.push_back(current_group);
            group_starts.push_back(i);
        }
    }
    group_ends.push_back(n);
}

void TopDownBVHBuilder::build_with_groups(BVH& bvh, const vec3* lowers, const vec3* uppers, const int* groups, int n)
{
    // Breakdown of building with groups:
    // 1. Sort and compute group ranges, finding the start and end indices of each group. Since the
    // primitives are not guaranteed to be sorted by group, we sort them by group id first.
    // 2. Find the bounds of each group
    // 3. Build group-level BVH, splitting based on group ids
    // 4. Expand each group leaf into a per-primitive subtree in ascending group order
    // 5. Reorder the tree so that all the leaf nodes are stored in the front

    // 1. Sort and compute group ranges
    std::vector<int> unique_groups, group_starts, group_ends, sorted_indices;
    compute_group_ranges(groups, n, unique_groups, group_starts, group_ends, sorted_indices);

    for (int i = 0; i < n; ++i) {
        int prim = sorted_indices[i];
        bvh.primitive_indices[i] = prim;
    }

    const int num_groups = int(unique_groups.size());

    std::vector<int> group_indices(num_groups);
    std::vector<int> group_leaf_node_index(num_groups, -1);
    std::vector<vec3> group_lowers(num_groups), group_uppers(num_groups);

    // 2. Find the bounds of each group
    for (int g = 0; g < num_groups; ++g) {
        group_indices[g] = g;
        bounds3 gb;
        for (int i = group_starts[g]; i < group_ends[g]; ++i) {
            int prim = bvh.primitive_indices[i];
            gb.add_bounds(lowers[prim], uppers[prim]);
        }
        group_lowers[g] = gb.lower;
        group_uppers[g] = gb.upper;
    }

    bvh.num_nodes = 0;
    bvh.num_leaf_nodes = 0;

    // 3. Build the group-level BVH, storing group ranges in leaf nodes
    std::function<int(int, int, int, int)> build_groups = [&](int start, int end, int depth, int parent) -> int {
        const int count = end - start;
        const int node_index = bvh.num_nodes++;
        if (depth > bvh.max_depth)
            bvh.max_depth = depth;

        bounds3 b;
        for (int i = start; i < end; ++i) {
            int g = group_indices[i];
            b.add_bounds(group_lowers[g], group_uppers[g]);
        }

        if (count == 1) {
            // leaf holds the contiguous primitive range for this group
            int g = group_indices[start];
            int prim_start = group_starts[g];
            int prim_end = group_ends[g];
            bvh.node_lowers[node_index] = make_node(b.lower, prim_start, true);
            bvh.node_uppers[node_index] = make_node(b.upper, prim_end, false);
            bvh.node_parents[node_index] = parent;
            group_leaf_node_index[g] = node_index;
            return node_index;
        }

        // then median split on based on group ids
        int mid = (start + end) / 2;
        std::nth_element(
            group_indices.begin() + start, group_indices.begin() + mid, group_indices.begin() + end,
            [&](int a, int b) { return unique_groups[a] < unique_groups[b]; }
        );

        int left = build_groups(start, mid, depth + 1, node_index);
        int right = build_groups(mid, end, depth + 1, node_index);

        bvh.node_lowers[node_index] = make_node(b.lower, left, false);
        bvh.node_uppers[node_index] = make_node(b.upper, right, false);
        bvh.node_parents[node_index] = parent;
        return node_index;
    };

    for (int i = 0; i < num_groups; ++i)
        group_indices[i] = i;
    *bvh.root = build_groups(0, num_groups, 0, -1);

    // 4. Expand each group leaf into a per-primitive subtree in ascending group order
    for (int g = 0; g < num_groups; ++g) {
        int node = group_leaf_node_index[g];
        if (node < 0)
            continue;
        int prim_start = bvh.node_lowers[node].i;
        int prim_end = bvh.node_uppers[node].i;
        // Replace this packed leaf with a full subtree over [prim_start, prim_end)
        bvh.node_lowers[node].b = 0;
        build_recursive(bvh, lowers, uppers, prim_start, prim_end, 0, bvh.node_parents[node], node);
    }

    // 5. Reorder the tree so that all the leaf nodes are stored in the front
    reorder_top_down_bvh(bvh);
}


void TopDownBVHBuilder::build(
    BVH& bvh, const vec3* lowers, const vec3* uppers, int n, int in_constructor_type, int* groups
)
{
    assert(n >= 0);
    if (n > 0) {
        assert(lowers != nullptr && uppers != nullptr && "Pointers must be valid for n > 0");
    }

    constructor_type = in_constructor_type;
    if (constructor_type != BVH_CONSTRUCTOR_SAH && constructor_type != BVH_CONSTRUCTOR_MEDIAN) {
        fprintf(
            stderr,
            "Unrecognized Constructor type: %d! For CPU constructor it should be either SAH (%d) or Median (%d)!\n",
            constructor_type, BVH_CONSTRUCTOR_SAH, BVH_CONSTRUCTOR_MEDIAN
        );
        return;
    }

    if (n < 0) {
        fprintf(stderr, "Error: Cannot build BVH with a negative primitive count: %d\n", n);
        initialize_empty(bvh);
        return;
    } else if (n == 0) {
        initialize_empty(bvh);
        return;
    } else if (n > INT_MAX / 2) {
        fprintf(stderr, "Error: Primitive count %d is too large and would cause an integer overflow.\n", n);
        initialize_empty(bvh);
        return;
    }

    bvh.max_depth = 0;
    bvh.max_nodes = 2 * n - 1;

    bvh.node_lowers
        = static_cast<BVHPackedNodeHalf*>(wp_alloc_host(sizeof(BVHPackedNodeHalf) * bvh.max_nodes, "(native:bvh)"));
    bvh.node_uppers
        = static_cast<BVHPackedNodeHalf*>(wp_alloc_host(sizeof(BVHPackedNodeHalf) * bvh.max_nodes, "(native:bvh)"));
    bvh.node_parents = static_cast<int*>(wp_alloc_host(sizeof(int) * bvh.max_nodes, "(native:bvh)"));
    bvh.node_counts = nullptr;
    bvh.num_items = n;

    bvh.root = static_cast<int*>(wp_alloc_host(sizeof(int), "(native:bvh)"));
    bvh.root[0] = 0;

    bvh.primitive_indices = static_cast<int*>(wp_alloc_host(sizeof(int) * n, "(native:bvh)"));
    for (int i = 0; i < n; ++i)
        bvh.primitive_indices[i] = i;

    if (groups) {
        build_with_groups(bvh, lowers, uppers, groups, n);
    } else {
        build_recursive(bvh, lowers, uppers, 0, n, 0, -1);
    }
}

void TopDownBVHBuilder::rebuild(BVH& bvh, int in_constructor_type)
{
    if (in_constructor_type != BVH_CONSTRUCTOR_SAH && in_constructor_type != BVH_CONSTRUCTOR_MEDIAN) {
        fprintf(
            stderr,
            "Unrecognized Constructor type: %d! For CPU constructor it should be either SAH (%d) or Median (%d)!\n",
            in_constructor_type, BVH_CONSTRUCTOR_SAH, BVH_CONSTRUCTOR_MEDIAN
        );
        return;
    }
    if (bvh.num_items == 0)
        return;

    constructor_type = in_constructor_type;
    for (int i = 0; i < bvh.num_items; ++i)
        bvh.primitive_indices[i] = i;

    bvh.max_depth = 0;
    bvh.num_nodes = 0;
    bvh.num_leaf_nodes = 0;

    if (bvh.item_groups) {
        build_with_groups(bvh, bvh.item_lowers, bvh.item_uppers, bvh.item_groups, bvh.num_items);
    } else {
        build_recursive(bvh, bvh.item_lowers, bvh.item_uppers, 0, bvh.num_items, 0, -1);
    }
}


bounds3 TopDownBVHBuilder::calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end)
{
    bounds3 u;

    for (int i = start; i < end; ++i) {
        u.add_bounds(lowers[indices[i]], uppers[indices[i]]);
    }

    return u;
}

struct PartitionPredicateMedian {
    PartitionPredicateMedian(const vec3* lowers, const vec3* uppers, int a)
        : lowers(lowers)
        , uppers(uppers)
        , axis(a)
    {
    }

    bool operator()(int a, int b) const
    {
        vec3 a_center = 0.5f * (lowers[a] + uppers[a]);
        vec3 b_center = 0.5f * (lowers[b] + uppers[b]);

        return a_center[axis] < b_center[axis];
    }

    const vec3* lowers;
    const vec3* uppers;
    int axis;
};


int TopDownBVHBuilder::partition_median(
    const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds
)
{
    assert(end - start >= 2);

    vec3 edges = range_bounds.edges();

    int axis = longest_axis(edges);

    const int k = (start + end) / 2;

    std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(lowers, uppers, axis));

    return k;
}

struct PartitionPredicateMidPoint {
    PartitionPredicateMidPoint(const vec3* lowers, const vec3* uppers, int a, float m)
        : lowers(lowers)
        , uppers(uppers)
        , axis(a)
        , mid(m)
    {
    }

    bool operator()(int index) const
    {
        vec3 center = 0.5f * (lowers[index] + uppers[index]);

        return center[axis] <= mid;
    }

    const vec3* lowers;
    const vec3* uppers;

    int axis;
    float mid;
};


int TopDownBVHBuilder::partition_midpoint(
    const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds
)
{
    assert(end - start >= 2);

    vec3 edges = range_bounds.edges();
    vec3 center = range_bounds.center();

    int axis = longest_axis(edges);
    float mid = center[axis];

    int* upper = std::partition(indices + start, indices + end, PartitionPredicateMidPoint(lowers, uppers, axis, mid));

    int k = upper - indices;

    // if we failed to split items then just split in the middle
    if (k == start || k == end)
        k = (start + end) / 2;

    return k;
}

float TopDownBVHBuilder::partition_sah_indices(
    const vec3* lowers,
    const vec3* uppers,
    const int* indices,
    int start,
    int end,
    bounds3 range_bounds,
    int& split_axis
)
{
    int buckets_counts[SAH_NUM_BUCKETS];
    bounds3 buckets[SAH_NUM_BUCKETS];
    float left_areas[SAH_NUM_BUCKETS - 1];
    float right_areas[SAH_NUM_BUCKETS - 1];

    assert(end - start >= 2);

    int n = end - start;

    bounds3 centroid_bounds;
    for (int i = start; i < end; ++i) {
        vec3 item_center = 0.5f * (lowers[indices[i]] + uppers[indices[i]]);
        centroid_bounds.add_point(item_center);
    }
    vec3 edges = centroid_bounds.edges();

    split_axis = longest_axis(edges);

    // compute each bucket
    float range_start = centroid_bounds.lower[split_axis];
    float range_end = centroid_bounds.upper[split_axis];

    // Guard against zero extent along the split axis to avoid division-by-zero
    if (range_end <= range_start) {
        // Returning range_start will cause the caller's partition to produce an empty left side;
        // the recursive builder already falls back to a midpoint split in that case.
        return range_start;
    }

    std::fill(buckets_counts, buckets_counts + SAH_NUM_BUCKETS, 0);
    for (int item_idx = start; item_idx < end; item_idx++) {
        vec3 item_center = 0.5f * (lowers[indices[item_idx]] + uppers[indices[item_idx]]);
        int bucket_idx = SAH_NUM_BUCKETS * (item_center[split_axis] - range_start) / (range_end - range_start);
        // clamp into valid range [0, SAH_NUM_BUCKETS-1]
        if (bucket_idx < 0)
            bucket_idx = 0;
        if (bucket_idx >= SAH_NUM_BUCKETS)
            bucket_idx = SAH_NUM_BUCKETS - 1;

        bounds3 item_bound(lowers[indices[item_idx]], uppers[indices[item_idx]]);

        if (buckets_counts[bucket_idx]) {
            buckets[bucket_idx] = bounds_union(item_bound, buckets[bucket_idx]);
        } else {
            buckets[bucket_idx] = item_bound;
        }

        buckets_counts[bucket_idx]++;
    }

    bounds3 left;
    bounds3 right;

    // n - 1 division points for n buckets
    int counts_l[SAH_NUM_BUCKETS - 1];
    int counts_r[SAH_NUM_BUCKETS - 1];

    int count_l = 0;
    int count_r = 0;
    // build cumulative bounds and area from left and right
    for (int i = 0; i < SAH_NUM_BUCKETS - 1; ++i) {
        bounds3 bound_start = buckets[i];
        bounds3 bound_end = buckets[SAH_NUM_BUCKETS - i - 1];

        left = bounds_union(left, bound_start);
        right = bounds_union(right, bound_end);

        left_areas[i] = left.area();
        right_areas[SAH_NUM_BUCKETS - i - 2] = right.area();

        count_l += buckets_counts[i];
        count_r += buckets_counts[SAH_NUM_BUCKETS - i - 1];

        counts_l[i] = count_l;
        counts_r[SAH_NUM_BUCKETS - i - 2] = count_r;
    }

    float invTotalArea = 1.0f / range_bounds.area();

    // find split point i that minimizes area(left[i]) * count[left[i]] + area(right[i]) * count[right[i]]
    int minSplit = 0;
    float minCost = FLT_MAX;
    for (int i = 0; i < SAH_NUM_BUCKETS - 1; ++i) {
        float pBelow = left_areas[i] * invTotalArea;
        float pAbove = right_areas[i] * invTotalArea;

        float cost = pBelow * counts_l[i] + pAbove * counts_r[i];

        if (cost < minCost) {
            minCost = cost;
            minSplit = i;
        }
    }

    // return the dividing
    assert(minSplit >= 0 && minSplit < SAH_NUM_BUCKETS - 1);
    float split_point = range_start + (minSplit + 1) * (range_end - range_start) / SAH_NUM_BUCKETS;

    return split_point;
}


int TopDownBVHBuilder::build_recursive(
    BVH& bvh, const vec3* lowers, const vec3* uppers, int start, int end, int depth, int parent, int assigned_node
)
{
    // Build subtree over [start,end). If assigned_node >= 0,
    // reuse that node index instead of allocating a new one.
    assert(start < end);

    const int n = end - start;
    const int node_index = (assigned_node >= 0) ? assigned_node : bvh.num_nodes++;

    if (assigned_node < 0)
        assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth)
        bvh.max_depth = depth;

    bounds3 b = calc_bounds(lowers, uppers, bvh.primitive_indices, start, end);

    // If the depth exceeds BVH_QUERY_STACK_SIZE, an out-of-bounds access bug may occur during querying.
    // In that case, we merge the following nodes into a single large leaf node.
    if (n <= bvh.leaf_size || depth >= BVH_QUERY_STACK_SIZE) {
        bvh.node_lowers[node_index] = make_node(b.lower, start, true);
        bvh.node_uppers[node_index] = make_node(b.upper, end, false);
        bvh.node_parents[node_index] = parent;
        bvh.num_leaf_nodes++;
        return node_index;
    }

    int split = -1;
    if (constructor_type == BVH_CONSTRUCTOR_SAH)
    // SAH constructor
    {
        int split_axis = -1;
        float split_point = partition_sah_indices(lowers, uppers, bvh.primitive_indices, start, end, b, split_axis);
        auto boundary = std::partition(bvh.primitive_indices + start, bvh.primitive_indices + end, [&](int i) {
            return 0.5f * (lowers[i] + uppers[i])[split_axis] < split_point;
        });

        split = std::distance(bvh.primitive_indices + start, boundary) + start;
    } else if (constructor_type == BVH_CONSTRUCTOR_MEDIAN)
    // Median constructor
    {
        split = partition_median(lowers, uppers, bvh.primitive_indices, start, end, b);
    } else {
        printf("Unknown type of BVH constructor: %d!\n", constructor_type);
        return -1;
    }

    if (split == start || split == end) {
        // partitioning failed, split down the middle
        split = (start + end) / 2;
    }

    int left_child = build_recursive(bvh, lowers, uppers, start, split, depth + 1, node_index);
    int right_child = build_recursive(bvh, lowers, uppers, split, end, depth + 1, node_index);

    bvh.node_lowers[node_index] = make_node(b.lower, left_child, false);
    bvh.node_uppers[node_index] = make_node(b.upper, right_child, false);
    bvh.node_parents[node_index] = parent;
    return node_index;
}


void reorder_top_down_bvh(BVH& bvh_host)
{
    // reorder bvh_host such that its nodes are in the front
    // this is essential for the device refit
    BVHPackedNodeHalf* node_lowers_reordered = static_cast<BVHPackedNodeHalf*>(
        wp_alloc_host(sizeof(BVHPackedNodeHalf) * bvh_host.max_nodes, "(native:bvh)")
    );
    BVHPackedNodeHalf* node_uppers_reordered = static_cast<BVHPackedNodeHalf*>(
        wp_alloc_host(sizeof(BVHPackedNodeHalf) * bvh_host.max_nodes, "(native:bvh)")
    );

    int* node_parents_reordered = static_cast<int*>(wp_alloc_host(sizeof(int) * bvh_host.max_nodes, "(native:bvh)"));

    std::vector<int> old_to_new(bvh_host.max_nodes, -1);

    // We will place nodes in this order:
    //   Pass 1: leaf nodes (except if it's the root index)
    //   Pass 2: non-leaf, non-root
    //   Pass 3: root node
    int next_pos = 0;

    const int root_index = *bvh_host.root;
    // Pass 1: place leaf nodes at the front
    // During grouped build, it is possible that some leaf nodes do
    // not satisfy the ascending group id assumption. This would happen if
    // a specific group has <= leaf size primitives, and so the leaf node is
    // built in-place rather than on the node frontier like the other leaves.
    // During this pass we can detect if this has occurred and call a sort to fix it.
    bool needs_sort = false;
    int prev_group = INT_MIN;
    for (int i = 0; i < bvh_host.num_nodes; ++i) {
        if (bvh_host.node_lowers[i].b) {
            if (bvh_host.item_groups) {
                int group = bvh_host.item_groups[bvh_host.primitive_indices[bvh_host.node_lowers[i].i]];
                if (group < prev_group)
                    needs_sort = true;
                prev_group = group;
            }
            node_lowers_reordered[next_pos] = bvh_host.node_lowers[i];
            node_uppers_reordered[next_pos] = bvh_host.node_uppers[i];
            old_to_new[i] = next_pos;
            next_pos++;
        }
    }

    if (needs_sort) {
        const int* groups = bvh_host.item_groups;

        std::vector<int> leaf_indices;
        leaf_indices.reserve(next_pos);
        for (int i = 0; i < bvh_host.num_nodes; ++i) {
            if (bvh_host.node_lowers[i].b) {
                leaf_indices.push_back(i);
            }
        }

        std::stable_sort(leaf_indices.begin(), leaf_indices.end(), [&](int a, int b) {
            int ga = groups[bvh_host.primitive_indices[bvh_host.node_lowers[a].i]];
            int gb = groups[bvh_host.primitive_indices[bvh_host.node_lowers[b].i]];
            if (ga == gb)
                return a < b;
            return ga < gb;
        });

        next_pos = 0;
        for (size_t i = 0; i < leaf_indices.size(); ++i) {
            int old_index = leaf_indices[i];
            node_lowers_reordered[next_pos] = bvh_host.node_lowers[old_index];
            node_uppers_reordered[next_pos] = bvh_host.node_uppers[old_index];
            old_to_new[old_index] = next_pos;
            next_pos++;
        }
    }
    bvh_host.num_leaf_nodes = next_pos;

    // Pass 2: place non-leaf, non-root nodes
    for (int i = 0; i < bvh_host.num_nodes; ++i) {
        if (i == root_index) {
            if (bvh_host.node_lowers[i].b)  // if root node is a leaf node, there must be only be one node
            {
                *bvh_host.root = 0;
            } else {
                *bvh_host.root = next_pos;
            }
        }
        if (!bvh_host.node_lowers[i].b) {
            node_lowers_reordered[next_pos] = bvh_host.node_lowers[i];
            node_uppers_reordered[next_pos] = bvh_host.node_uppers[i];
            old_to_new[i] = next_pos;
            next_pos++;
        }
    }

    // We can do that by enumerating all old->new pairs:
    for (int old_index = 0; old_index < bvh_host.num_nodes; ++old_index) {
        int new_index = old_to_new[old_index];  // new index

        int old_parent = bvh_host.node_parents[old_index];
        if (old_parent != -1) {
            node_parents_reordered[new_index] = old_to_new[old_parent];
        } else {
            node_parents_reordered[new_index] = -1;
        }

        // only need to modify the child index of non-leaf nodes
        if (!bvh_host.node_lowers[old_index].b) {
            node_lowers_reordered[new_index].i = old_to_new[bvh_host.node_lowers[old_index].i];
            node_uppers_reordered[new_index].i = old_to_new[bvh_host.node_uppers[old_index].i];
        }
    }

    wp_free_host(bvh_host.node_lowers);
    wp_free_host(bvh_host.node_uppers);
    wp_free_host(bvh_host.node_parents);

    bvh_host.node_lowers = node_lowers_reordered;
    bvh_host.node_uppers = node_uppers_reordered;
    bvh_host.node_parents = node_parents_reordered;
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Host LBVH: a sequential execution of the CUDA LBVH builder (LinearBVHBuilderGPU::build in
// bvh.cu) and of its refit kernel. Used for Metal trees, whose host-built arrays are read by
// Metal kernels: the tree (node order, child indices, leaf packing, parents, primitive order and
// node bounds) is bitwise the one the CUDA build produces from the same item bounds.
//
// Why sequential execution is exact: every CUDA kernel of the build is a pure function of its
// thread index and of data that is fixed before the kernel runs, except build_hierarchy and the
// refit, whose atomics only decide which of a node's two children goes on to process the parent;
// the parent's contents (child slots written by each child, range, bounds) do not depend on it.
// The key sort is CUB's radix sort of (key, index) pairs, which is stable; a stable sort by key of
// the identity permutation gives the same order.
//
// Floating point: the only arithmetic is the Morton quantization (0.5f * (lower + upper),
// (center - grid_lower) * inv_edges, x * 1024, edges + 0.0001f, 1.0f / edges) and min/max. None of
// these is a multiply feeding an add, so nvcc's --fmad=true has nothing to fuse; contraction is
// also switched off here so the host compiler cannot fuse either. Min/max follow the CUDA fminf /
// fmaxf semantics (NaN is missing data) where the kernels use wp::min/max, and bounds3::add_bounds
// where they use it. Equal zeros of opposite sign are ordered -0.0 < +0.0, taken to be what the
// GPU min/max instructions return; this is the one assumption not verified against a CUDA run, and
// it matters only for bounds holding signed zeros.

namespace {

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

inline float lbvh_fmin(float a, float b)
{
    if (a != a)
        return b;
    if (b != b)
        return a;
    if (a == b)
        return std::signbit(a) ? a : b;
    return (a < b) ? a : b;
}

inline float lbvh_fmax(float a, float b)
{
    if (a != a)
        return b;
    if (b != b)
        return a;
    if (a == b)
        return std::signbit(a) ? b : a;
    return (a > b) ? a : b;
}

inline vec3 lbvh_min(const vec3& a, const vec3& b)
{
    return vec3(lbvh_fmin(a[0], b[0]), lbvh_fmin(a[1], b[1]), lbvh_fmin(a[2], b[2]));
}

inline vec3 lbvh_max(const vec3& a, const vec3& b)
{
    return vec3(lbvh_fmax(a[0], b[0]), lbvh_fmax(a[1], b[1]), lbvh_fmax(a[2], b[2]));
}

inline vec3 node_bound(const BVHPackedNodeHalf& n) { return vec3(n.x, n.y, n.z); }

inline void set_node_bound(BVHPackedNodeHalf& n, const vec3& b)
{
    n.x = b[0];
    n.y = b[1];
    n.z = b[2];
}

inline int lbvh_clz64(uint64_t x) { return x == 0 ? 64 : __builtin_clzll(x); }

// Morton key of one item (bvh.cu: compute_morton_codes)
inline uint64_t lbvh_key(
    const vec3& lower, const vec3& upper, const vec3& grid_lower, const vec3& grid_inv_edges, const int* groups, int i
)
{
    vec3 center = 0.5f * (lower + upper);
    vec3 local = cw_mul((center - grid_lower), grid_inv_edges);
    const uint64_t morton_code = static_cast<uint64_t>(morton3<1024>(local[0], local[1], local[2]));
    const uint64_t group = groups ? static_cast<uint32_t>(groups[i]) : 0u;
    return (group << 32) | morton_code;
}

}  // anonymous namespace

// Builds an LBVH into bvh's existing arrays (node_lowers, node_uppers, node_parents of max_nodes = 2n - 1
// entries, primitive_indices of n, root); reads bvh.item_lowers/uppers/groups and bvh.leaf_size.
void lbvh_build_host(BVH& bvh)
{
    const int n = bvh.num_items;
    if (n <= 0)
        return;
    const vec3* item_lowers = bvh.item_lowers;
    const vec3* item_uppers = bvh.item_uppers;
    const int* item_groups = bvh.item_groups;
    const int max_nodes = 2 * n - 1;

    // compute_total_bounds (from +/-FLT_MAX) and compute_total_inv_edges
    vec3 total_lower(FLT_MAX);
    vec3 total_upper(-FLT_MAX);
    for (int i = 0; i < n; ++i) {
        total_lower = lbvh_min(total_lower, item_lowers[i]);
        total_upper = lbvh_max(total_upper, item_uppers[i]);
    }
    vec3 edges = (total_upper - total_lower);
    edges += vec3(0.0001f);
    const vec3 inv_edges = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);

    // compute_morton_codes, then the (stable) radix sort of (key, index) pairs
    std::vector<uint64_t> unsorted_keys(n);
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        unsorted_keys[i] = lbvh_key(item_lowers[i], item_uppers[i], total_lower, inv_edges, item_groups, i);
        indices[i] = i;
    }
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) { return unsorted_keys[a] < unsorted_keys[b]; });
    std::vector<uint64_t> keys(n);
    for (int i = 0; i < n; ++i) {
        keys[i] = unsorted_keys[indices[i]];
        bvh.primitive_indices[i] = indices[i];
    }

    // compute_key_deltas
    std::vector<int> deltas(n);
    for (int i = 0; i < n - 1; ++i)
        deltas[i] = lbvh_clz64(keys[i] ^ keys[i + 1]);

    // build_leaves
    std::vector<int> range_lefts(max_nodes), range_rights(max_nodes), num_children(max_nodes, 0);
    BVHPackedNodeHalf* lowers = bvh.node_lowers;
    BVHPackedNodeHalf* uppers = bvh.node_uppers;
    int* parents = bvh.node_parents;
    for (int i = 0; i < n; ++i) {
        const int item = indices[i];
        lowers[i] = make_node(item_lowers[item], i, true);
        uppers[i] = make_node(item_uppers[item], i, false);
        range_lefts[i] = i;
        range_rights[i] = i;
    }

    // build_hierarchy, one leaf "thread" after the other
    const int* primitive_indices = bvh.primitive_indices;
    const int internal_offset = n;
    for (int thread = 0; thread < n; ++thread) {
        int index = thread;
        for (;;) {
            const int left = range_lefts[index];
            const int right = range_rights[index];

            if (left == 0 && right == n - 1) {
                *bvh.root = index;
                parents[index] = -1;
                break;
            }

            bool parent_right = false;
            if (left == 0) {
                parent_right = true;
            } else {
                bool decided = false;
                const uint32_t group_left = (uint32_t)(keys[left] >> 32);
                const uint32_t group_right = (uint32_t)(keys[right] >> 32);
                if (group_left == group_right) {
                    const uint32_t group = group_left;
                    const bool can_go_right_same = (right < n - 1) && ((uint32_t)(keys[right + 1] >> 32) == group);
                    const bool can_go_left_same = ((uint32_t)(keys[left - 1] >> 32) == group);
                    if (can_go_right_same ^ can_go_left_same) {
                        parent_right = can_go_right_same;
                        decided = true;
                    }
                }
                if (!decided) {
                    if (right != n - 1 && deltas[right] >= deltas[left - 1]) {
                        if (deltas[right] == deltas[left - 1])
                            parent_right = (primitive_indices[left - 1] % 2) ^ (primitive_indices[right] % 2);
                        else
                            parent_right = true;
                    } else {
                        parent_right = false;
                    }
                }
            }

            int parent;
            if (parent_right) {
                parent = right + internal_offset;
                parents[index] = parent;
                lowers[parent].i = index;
                range_lefts[parent] = left;
            } else {
                parent = left + internal_offset - 1;
                parents[index] = parent;
                uppers[parent].i = index;
                range_rights[parent] = right;
            }

            if (num_children[parent]++ == 1) {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;
                const vec3 lower = lbvh_min(node_bound(lowers[left_child]), node_bound(lowers[right_child]));
                const vec3 upper = lbvh_max(node_bound(uppers[left_child]), node_bound(uppers[right_child]));
                lowers[parent] = make_node(lower, left_child, false);
                uppers[parent] = make_node(upper, right_child, false);
                index = parent;
            } else {
                break;
            }
        }
    }

    // mark_packed_leaf_nodes
    for (int node_index = 0; node_index < max_nodes; ++node_index) {
        int depth = 1;
        int parent = parents[node_index];
        while (parent != -1) {
            parent = parents[parent];
            depth++;
        }
        const int left = range_lefts[node_index];
        const int right = range_rights[node_index] + 1;
        const bool single_group = (keys[left] >> 32) == (keys[right - 1] >> 32);
        if (single_group && (right - left <= bvh.leaf_size || depth >= BVH_QUERY_STACK_SIZE)) {
            lowers[node_index].b = 1;
            lowers[node_index].i = left;
            uppers[node_index].i = right;
        }
    }

    bvh.max_nodes = max_nodes;
    bvh.num_nodes = max_nodes;
    bvh.num_leaf_nodes = n;
    bvh.max_depth = 0;
    bvh.constructor_type = BVH_CONSTRUCTOR_LBVH;
}

// Allocates and builds an LBVH (bvh_create_device's LBVH branch in bvh.cu).
void lbvh_create_host(vec3* lowers, vec3* uppers, int num_items, int* groups, int leaf_size, BVH& bvh)
{
    memset(&bvh, 0, sizeof(BVH));
    bvh.item_lowers = lowers;
    bvh.item_uppers = uppers;
    bvh.item_groups = groups;
    bvh.num_items = num_items;
    bvh.leaf_size = leaf_size;
    bvh.constructor_type = BVH_CONSTRUCTOR_LBVH;
    if (num_items <= 0) {
        bvh.num_items = 0;
        return;
    }
    if (num_items > INT_MAX / 2) {
        fprintf(stderr, "Error: Primitive count %d is too large and would cause an integer overflow.\n", num_items);
        bvh.num_items = 0;
        return;
    }
    const int max_nodes = 2 * num_items - 1;
    bvh.max_nodes = max_nodes;
    bvh.node_lowers
        = static_cast<BVHPackedNodeHalf*>(wp_alloc_host(sizeof(BVHPackedNodeHalf) * max_nodes, "(native:bvh)"));
    bvh.node_uppers
        = static_cast<BVHPackedNodeHalf*>(wp_alloc_host(sizeof(BVHPackedNodeHalf) * max_nodes, "(native:bvh)"));
    bvh.node_parents = static_cast<int*>(wp_alloc_host(sizeof(int) * max_nodes, "(native:bvh)"));
    bvh.primitive_indices = static_cast<int*>(wp_alloc_host(sizeof(int) * num_items, "(native:bvh)"));
    bvh.root = static_cast<int*>(wp_alloc_host(sizeof(int), "(native:bvh)"));
    memset(bvh.node_lowers, 0, sizeof(BVHPackedNodeHalf) * max_nodes);
    memset(bvh.node_uppers, 0, sizeof(BVHPackedNodeHalf) * max_nodes);
    *bvh.root = 0;
    lbvh_build_host(bvh);
}

// bvh_refit_kernel in bvh.cu, one leaf "thread" after the other. Like the kernel it leaves the bounds of
// nodes muted by leaf packing as they are (no query reaches them).
void lbvh_refit_host(BVH& bvh)
{
    const int num_leaves = bvh.num_leaf_nodes;
    if (!bvh.node_lowers || num_leaves <= 0)
        return;
    BVHPackedNodeHalf* lowers = bvh.node_lowers;
    BVHPackedNodeHalf* uppers = bvh.node_uppers;
    const int* parents = bvh.node_parents;

    auto leaf_bounds = [&](int node) {
        bounds3 bound;
        for (int c = lowers[node].i; c < int(uppers[node].i); c++) {
            const int primitive = bvh.primitive_indices[c];
            bound.add_bounds(bvh.item_lowers[primitive], bvh.item_uppers[primitive]);
        }
        set_node_bound(lowers[node], bound.lower);
        set_node_bound(uppers[node], bound.upper);
    };

    std::vector<int> child_count(bvh.max_nodes, 0);
    for (int thread = 0; thread < num_leaves; ++thread) {
        int index = thread;
        if (!lowers[index].b)
            continue;
        int parent = parents[index];
        if (parent == -1 || !lowers[parent].b)
            leaf_bounds(index);

        for (;;) {
            parent = parents[index];
            if (parent == -1)
                break;
            if (child_count[parent]++ != 1)
                break;
            if (lowers[parent].b) {
                const int parent_parent = parents[parent];
                if (parent_parent == -1 || !lowers[parent_parent].b)
                    leaf_bounds(parent);
            } else {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;
                set_node_bound(
                    lowers[parent], lbvh_min(node_bound(lowers[left_child]), node_bound(lowers[right_child]))
                );
                set_node_bound(
                    uppers[parent], lbvh_max(node_bound(uppers[left_child]), node_bound(uppers[right_child]))
                );
            }
            index = parent;
        }
    }
}

// Triangle bounds as compute_triangle_bounds in mesh.cu computes them (min(min(p, q), r) with the CUDA
// min/max semantics), for meshes whose BVH is an LBVH.
void lbvh_triangle_bounds(const vec3& p, const vec3& q, const vec3& r, vec3& lower, vec3& upper)
{
    lower = lbvh_min(lbvh_min(p, q), r);
    upper = lbvh_max(lbvh_max(p, q), r);
}

#if defined(__clang__)
#pragma clang fp contract(on)
#endif


void bvh_refit_recursive(BVH& bvh, int index)
{
    BVHPackedNodeHalf& lower = bvh.node_lowers[index];
    BVHPackedNodeHalf& upper = bvh.node_uppers[index];

    if (lower.b) {
        // update leaf from items
        bounds3 bound;
        for (int item_counter = lower.i; item_counter < upper.i; item_counter++) {
            const int item = bvh.primitive_indices[item_counter];
            bound.add_bounds(bvh.item_lowers[item], bvh.item_uppers[item]);
        }

        reinterpret_cast<vec3&>(lower) = bound.lower;
        reinterpret_cast<vec3&>(upper) = bound.upper;
    } else {
        int left_index = lower.i;
        int right_index = upper.i;

        bvh_refit_recursive(bvh, left_index);
        bvh_refit_recursive(bvh, right_index);

        // compute union of children
        const vec3& left_lower = reinterpret_cast<const vec3&>(bvh.node_lowers[left_index]);
        const vec3& left_upper = reinterpret_cast<const vec3&>(bvh.node_uppers[left_index]);

        const vec3& right_lower = reinterpret_cast<const vec3&>(bvh.node_lowers[right_index]);
        const vec3& right_upper = reinterpret_cast<const vec3&>(bvh.node_uppers[right_index]);

        // union of child bounds
        vec3 new_lower = std_min(left_lower, right_lower);
        vec3 new_upper = std_max(left_upper, right_upper);

        // write new BVH nodes
        reinterpret_cast<vec3&>(lower) = new_lower;
        reinterpret_cast<vec3&>(upper) = new_upper;
    }
}

void bvh_refit_host(BVH& bvh)
{
    if (bvh.constructor_type == BVH_CONSTRUCTOR_LBVH) {
        lbvh_refit_host(bvh);
        return;
    }
    bvh_refit_recursive(bvh, *bvh.root);
}
void bvh_rebuild_host(BVH& bvh, int constructor_type)
{
    if (constructor_type == BVH_CONSTRUCTOR_CUBQL) {
        if (bvh.item_groups) {
            wp::set_error_string("Warp error: grouped BVHs are not supported with cuBQL construction");
            return;
        }
        if (bvh.constructor_type != BVH_CONSTRUCTOR_CUBQL) {
            wp::set_error_string("Warp error: cannot rebuild a non-cuBQL BVH with cuBQL construction in place");
            return;
        }
        cubql_bvh_rebuild_host(bvh);
        bvh.constructor_type = constructor_type;
        return;
    }

    if (bvh.constructor_type == BVH_CONSTRUCTOR_CUBQL) {
        wp::set_error_string("Warp error: cannot rebuild a cuBQL BVH with a different constructor in place");
        return;
    }

    if (constructor_type == BVH_CONSTRUCTOR_LBVH) {
        // in place, as bvh_rebuild_device in bvh.cu (the buffers of any non-cuBQL tree fit)
        lbvh_build_host(bvh);
        return;
    }

    TopDownBVHBuilder builder;
    builder.rebuild(bvh, constructor_type);
    bvh.constructor_type = constructor_type;
}

}  // namespace wp


// making the class accessible from python


namespace {
// host-side copy of bvh descriptors, maps GPU bvh address (id) to a CPU desc
std::map<uint64_t, BVH> g_bvh_descriptors;
}  // anonymous namespace


namespace wp {

bool bvh_get_descriptor(uint64_t id, BVH& bvh)
{
    const auto& iter = g_bvh_descriptors.find(id);
    if (iter == g_bvh_descriptors.end())
        return false;
    else
        bvh = iter->second;
    return true;
}

void bvh_add_descriptor(uint64_t id, const BVH& bvh) { g_bvh_descriptors[id] = bvh; }

void bvh_rem_descriptor(uint64_t id) { g_bvh_descriptors.erase(id); }


// create in-place given existing descriptor
void bvh_create_host(
    vec3* lowers, vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size, BVH& bvh
)
{
    if (constructor_type == BVH_CONSTRUCTOR_CUBQL) {
        if (groups) {
            wp::set_error_string("Warp error: grouped BVHs are not supported with cuBQL construction");
            memset(&bvh, 0, sizeof(BVH));
            return;
        }
        cubql_bvh_create_host(lowers, uppers, num_items, leaf_size, bvh);
        bvh.constructor_type = constructor_type;
        return;
    }

    if (constructor_type == BVH_CONSTRUCTOR_LBVH) {
        lbvh_create_host(lowers, uppers, num_items, groups, leaf_size, bvh);
        return;
    }

    memset(&bvh, 0, sizeof(BVH));

    bvh.item_lowers = lowers;
    bvh.item_uppers = uppers;
    bvh.item_groups = groups;
    bvh.num_items = num_items;
    bvh.leaf_size = leaf_size;
    bvh.constructor_type = constructor_type;

    TopDownBVHBuilder builder;
    builder.build(bvh, lowers, uppers, num_items, constructor_type, groups);
}


void bvh_destroy_host(BVH& bvh)
{
    wp_free_host(bvh.node_lowers);
    wp_free_host(bvh.node_uppers);
    wp_free_host(bvh.node_parents);
    wp_free_host(bvh.primitive_indices);
    wp_free_host(bvh.root);

    bvh.node_lowers = nullptr;
    bvh.node_uppers = nullptr;
    bvh.node_parents = nullptr;
    bvh.primitive_indices = nullptr;
    bvh.root = nullptr;

    bvh.max_nodes = 0;
    bvh.num_items = 0;
    bvh.constructor_type = BVH_CONSTRUCTOR_SAH;
}

}  // namespace wp

uint64_t wp_bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size)
{
    BVH* bvh = static_cast<BVH*>(wp_alloc_host(sizeof(BVH), "(native:bvh)"));
    memset(bvh, 0, sizeof(BVH));
    wp::bvh_create_host(lowers, uppers, num_items, constructor_type, groups, leaf_size, *bvh);

    if (!bvh->node_lowers && num_items > 0) {
        wp_free_host(bvh);
        return 0;
    }

    return (uint64_t)bvh;
}

// Record a host BVH update into the active APIC byte stream (CPU graph
// capture); returns true if it was recorded (and therefore should NOT execute
// eagerly). During capture the update is deferred so replay re-runs it against
// the replay-time lowers/uppers; executing it now would leave the tree frozen
// at the capture-time bounds when the graph is later replayed (GH-1665).
static bool apic_capture_bvh_refit_host(uint64_t id)
{
    APICState* state = wp_apic_get_recording_state();
    if (!state)
        return false;
    apic_record_bvh_refit(state, id);
    return true;
}

static bool apic_capture_bvh_rebuild_host(uint64_t id, int constructor_type)
{
    APICState* state = wp_apic_get_recording_state();
    if (!state)
        return false;
    apic_record_bvh_rebuild(state, id, constructor_type);
    return true;
}

void wp_bvh_refit_host(uint64_t id)
{
    if (apic_capture_bvh_refit_host(id))
        return;

    BVH* bvh = (BVH*)(id);
    wp::bvh_refit_host(*bvh);
}

void wp_bvh_rebuild_host(uint64_t id, int constructor_type)
{
    if (apic_capture_bvh_rebuild_host(id, constructor_type))
        return;

    BVH* bvh = (BVH*)(id);
    wp::bvh_rebuild_host(*bvh, constructor_type);
}

void wp_bvh_destroy_host(uint64_t id)
{
    BVH* bvh = (BVH*)(id);
    wp::bvh_destroy_host(*bvh);
    wp_free_host(bvh);
}


namespace wp {
// Copies a host-addressable tree (see wp_bvh_get_tree)
int bvh_copy_tree(
    const BVH& bvh, int* info, void* node_lowers, void* node_uppers, int* node_parents, int* primitive_indices
)
{
    if (info) {
        info[0] = bvh.max_nodes;
        info[1] = bvh.num_items;
        info[2] = bvh.root ? *bvh.root : -1;
        info[3] = bvh.num_leaf_nodes;
        info[4] = bvh.constructor_type;
        info[5] = bvh.leaf_size;
    }
    const size_t nodes = bvh.node_lowers ? size_t(bvh.max_nodes) : 0;
    const size_t items = bvh.primitive_indices ? size_t(bvh.num_items) : 0;
    if (node_lowers && nodes)
        memcpy(node_lowers, bvh.node_lowers, sizeof(BVHPackedNodeHalf) * nodes);
    if (node_uppers && nodes)
        memcpy(node_uppers, bvh.node_uppers, sizeof(BVHPackedNodeHalf) * nodes);
    if (node_parents && nodes && bvh.node_parents)
        memcpy(node_parents, bvh.node_parents, sizeof(int) * nodes);
    if (primitive_indices && items)
        memcpy(primitive_indices, bvh.primitive_indices, sizeof(int) * items);
    return 1;
}
}  // namespace wp

// defined with the Metal mesh registry in mesh.cpp (0 elsewhere)
uint64_t metal_mesh_host_id(uint64_t id);
uint64_t metal_bvh_host_id(uint64_t id);

int wp_bvh_get_tree(
    uint64_t id,
    int device,
    int is_mesh,
    int* info,
    void* node_lowers,
    void* node_uppers,
    int* node_parents,
    int* primitive_indices
)
{
    uint64_t host_id = id;
    if (device)
        host_id = is_mesh ? metal_mesh_host_id(id) : metal_bvh_host_id(id);
    if (!host_id)
        return 0;
    const BVH& bvh = is_mesh ? reinterpret_cast<const Mesh*>(host_id)->bvh : *reinterpret_cast<const BVH*>(host_id);
    return wp::bvh_copy_tree(bvh, info, node_lowers, node_uppers, node_parents, primitive_indices);
}

#if defined(__APPLE__)

// On Apple platforms the device is a Metal GPU: the tree is built and refitted by the host code
// above, in Metal memory, and kernels read a descriptor holding GPU addresses (see metal_host.h).
// LBVH trees are built by lbvh_build_host, a sequential execution of the CUDA LBVH kernels, so a
// Metal tree is bitwise the tree CUDA builds from the same bounds; in-place rebuilds are LBVH, as
// on CUDA.
namespace {
std::map<uint64_t, MetalMirror<BVH>> g_metal_bvhs;

bool metal_bvh_update(const MetalMirror<BVH>& m)
{
    MetalAddressTranslator gpu { m.ordinal };
    *m.descriptor = metal_bvh_descriptor(gpu, *reinterpret_cast<BVH*>(m.host_id));
    return gpu.ok;
}
}  // anonymous namespace

uint64_t wp_bvh_create_device(
    void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size
)
{
    const int ordinal = metal_context_ordinal(context);
    if (wp_metal_synchronize(ordinal) != 0)  // the bounds may still be written by GPU kernels
        return 0;
    ScopedMetalHostAlloc scope(ordinal);
    MetalMirror<BVH> m { ordinal, wp_bvh_create_host(lowers, uppers, num_items, constructor_type, groups, leaf_size),
                         nullptr };
    if (!m.host_id)
        return 0;
    m.descriptor = static_cast<BVH*>(wp_alloc_host(sizeof(BVH), "(native:bvh)"));
    const uint64_t id = wp_metal_gpu_address(ordinal, m.descriptor);
    if (!id || !metal_bvh_update(m)) {
        wp_bvh_destroy_host(m.host_id);
        wp_free_host(m.descriptor);
        return 0;
    }
    g_metal_bvhs[id] = m;
    return id;
}

uint64_t metal_bvh_host_id(uint64_t id)
{
    auto it = g_metal_bvhs.find(id);
    if (it == g_metal_bvhs.end())
        return 0;
    wp_metal_synchronize(it->second.ordinal);  // a recorded rebuild or refit may still be pending
    return it->second.host_id;
}

// Runs a host-side BVH update now, or records it so a graph replays it in order with its dispatches.
static void metal_bvh_apply(uint64_t id, std::function<void(const MetalMirror<BVH>&)> update)
{
    auto it = g_metal_bvhs.find(id);
    if (it == g_metal_bvhs.end())
        return;
    const int ordinal = it->second.ordinal;
    auto apply = [=]() {
        auto it = g_metal_bvhs.find(id);
        if (it == g_metal_bvhs.end())
            return false;
        ScopedMetalHostAlloc scope(ordinal);
        update(it->second);
        return metal_bvh_update(it->second);
    };
    if (wp_metal_capture_host_op(ordinal, apply))
        return;
    // the bounds may still be written by GPU kernels; this entry point returns void, so a failure is
    // handed to the next synchronize
    if (wp_metal_synchronize(ordinal) != 0 || !apply())
        wp_metal_defer_error(ordinal);
}

void wp_bvh_refit_device(uint64_t id)
{
    metal_bvh_apply(id, [](const MetalMirror<BVH>& m) { wp_bvh_refit_host(m.host_id); });
}

void wp_bvh_rebuild_device(uint64_t id)
{
    metal_bvh_apply(id, [](const MetalMirror<BVH>& m) {
        // bvh_rebuild_device in bvh.cu: cuBQL trees rebuild with cuBQL, every other tree becomes an LBVH
        const int constructor_type = reinterpret_cast<BVH*>(m.host_id)->constructor_type;
        wp_bvh_rebuild_host(
            m.host_id, constructor_type == BVH_CONSTRUCTOR_CUBQL ? BVH_CONSTRUCTOR_CUBQL : BVH_CONSTRUCTOR_LBVH
        );
    });
}

void wp_bvh_destroy_device(uint64_t id)
{
    auto it = g_metal_bvhs.find(id);
    if (it == g_metal_bvhs.end())
        return;
    ScopedMetalHostAlloc scope(it->second.ordinal);
    wp_bvh_destroy_host(it->second.host_id);
    wp_free_host(it->second.descriptor);
    g_metal_bvhs.erase(it);
}

#else

uint64_t metal_bvh_host_id(uint64_t id) { return 0; }

#endif  // __APPLE__

#if !defined(__APPLE__) && !WP_ENABLE_CUDA

// stubs for platforms without a GPU backend
uint64_t wp_bvh_create_device(
    void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size
)
{
    return 0;
}
void wp_bvh_refit_device(uint64_t id) { }
void wp_bvh_destroy_device(uint64_t id) { }
void wp_bvh_rebuild_device(uint64_t id) { }

#endif  // !WP_ENABLE_CUDA
