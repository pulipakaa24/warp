// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CUDA's float32 logf / expf / sinhf (libdevice __nv_logf / __nv_expf / __nv_sinhf, as NVRTC inlines them into
// kernels compiled without fast_math), bit for bit, for Metal. Warp's wp.log / wp.exp / wp.sinh call these C names:
// on CUDA they are libdevice's, on Metal they were the Metal library's, which round differently (over all 2^32
// inputs Metal's log equals CUDA's on about half of the finite inputs, exp and sinh on about 97 %). Newton's implicit
// MPM evaluates them in its plastic-flow (Jp) and hardening updates, so the difference was systematic, not rounding
// noise of the kind the L4 itself produces.
//
// The sequences are the PTX NVRTC emits for the three functions (CUDA 12.9 NVVM, Warp 1.16, sm_89): a fused
// multiply-add wherever the PTX has fma.rn, a single rounding for every other product and sum (no product feeds an
// add, so ptxas cannot contract further; products are written as fma(a, b, signed zero) so the Metal compiler cannot
// contract them either), and ex2.approx.ftz.f32 = MUFU.EX2 through Benjamin Carleton's sass-math model (MIT
// License, Copyright (c) 2020 Benjamin Carleton, see licenses/sass-math-LICENSE.txt). Subnormal inputs and results
// follow CUDA's non-ftz forms on the bit patterns (Apple GPUs flush float32 subnormals in arithmetic).
// Needs metal_cuda_fast_math.h (wp_cuda_rcp, wp_mufu_square_approx).

// MUFU.EX2 coefficients (sass-math's ex2 table, packed {C0, C1 | C2 << 17}).
// clang-format off
constant uint2 wp_mufu_ex2_packed[64] = {
    {0x2000002u, 0x03dc58b9u}, {0x2059349u, 0x03ea59b0u}, {0x20b361du, 0x03f45aaau}, {0x210e8a4u, 0x03fe5ba7u},
    {0x216ab11u, 0x040c5ca6u}, {0x21c7d88u, 0x04125da9u}, {0x222603bu, 0x041e5eaeu}, {0x2285358u, 0x04285fb6u},
    {0x22e570bu, 0x043a60c0u}, {0x2346b80u, 0x044461ceu}, {0x23a90e7u, 0x044e62dfu}, {0x240c774u, 0x045e63f2u},
    {0x2470f50u, 0x04686509u}, {0x24d68afu, 0x04706623u}, {0x253d3c2u, 0x0482673fu}, {0x25a50b8u, 0x048e685fu},
    {0x260dfc4u, 0x049a6982u}, {0x2678119u, 0x04a86aa8u}, {0x26e34e8u, 0x04b06bd2u}, {0x274fb69u, 0x04c26cfeu},
    {0x27bd4cbu, 0x04ce6e2eu}, {0x282c147u, 0x04dc6f61u}, {0x289c10du, 0x04e67098u}, {0x290d458u, 0x04f271d2u},
    {0x297fb5cu, 0x0502730fu}, {0x29f3650u, 0x050e7450u}, {0x2a6856cu, 0x051e7594u}, {0x2ade8e9u, 0x052a76dcu},
    {0x2b560fdu, 0x053c7827u}, {0x2bcede4u, 0x054a7976u}, {0x2c48fd9u, 0x055c7ac8u}, {0x2cc4712u, 0x05667c1fu},
    {0x2d413ceu, 0x05767d79u}, {0x2dbf64au, 0x05827ed7u}, {0x2e3eec0u, 0x05968038u}, {0x2ebfd6cu, 0x05a2819eu},
    {0x2f42290u, 0x05b48307u}, {0x2fc5e69u, 0x05c68474u}, {0x304b136u, 0x05d885e5u}, {0x30d1b35u, 0x05e2875bu},
    {0x3159caau, 0x05f688d4u}, {0x31e35d5u, 0x060a8a51u}, {0x326e6f9u, 0x06168bd3u}, {0x32fb056u, 0x06268d59u},
    {0x3389232u, 0x06388ee3u}, {0x3418cd1u, 0x064c9071u}, {0x34aa078u, 0x065a9204u}, {0x353cd6cu, 0x066c939bu},
    {0x35d13f6u, 0x06829536u}, {0x3667459u, 0x069496d6u}, {0x36feee0u, 0x06a2987bu}, {0x37983d5u, 0x06b49a24u},
    {0x383337fu, 0x06cc9bd1u}, {0x38cfe29u, 0x06da9d84u}, {0x396e41eu, 0x06ee9f3bu}, {0x3a0e5acu, 0x0700a0f7u},
    {0x3ab031fu, 0x071aa2b7u}, {0x3b53cc3u, 0x072aa47du}, {0x3bf92e9u, 0x073aa648u}, {0x3ca05dfu, 0x0754a817u},
    {0x3d495f7u, 0x0764a9ecu}, {0x3df4381u, 0x077eabc5u}, {0x3ea0eceu, 0x0792ada4u}, {0x3f4f833u, 0x07a8af88u}
};
// clang-format on

inline float wp_cuda_f(uint b) { return as_type<float>(b); }
inline uint wp_cuda_u(float f) { return as_type<uint>(f); }

// a * b rounded once, kept apart from any following add (the Metal compiler contracts a visible product).
inline float wp_cuda_mul(float a, float b)
{
    return metal::fma(a, b, wp_cuda_f((wp_cuda_u(a) ^ wp_cuda_u(b)) & 0x80000000u));
}

// ex2.approx.ftz.f32 (sass-math ptxm_ex2_sm5x: reduction to sign, integer part and a 23-bit fixed-point fraction,
// then MUFU.EX2's table interpolation; inputs below -126 are halved and the result squared). Subnormal inputs and
// outputs flush to zero.
inline float wp_cuda_ex2(float x)
{
    uint xb = wp_cuda_u(x);
    if ((xb & 0x7FFFFFFFu) > 0x7F800000u)
        return wp_cuda_f(0x7FFFFFFFu);
    if ((xb & 0x7F800000u) == 0u)
        xb &= 0x80000000u;
    float xf = wp_cuda_f(xb);
    bool square = false;
    if (xf < -126.0f) {
        xf *= 0.5f;
        square = true;
    }
    const uint sgn = wp_cuda_u(xf) & 0x80000000u;
    uint red;
    if (xf >= 128.0f) {
        red = 0x81u << 23;
    } else if (xf <= -128.0f) {
        red = 0x80000000u | (0x81u << 23);
    } else {
        const float ip = metal::trunc(xf);
        uint fb = wp_cuda_u(xf - ip);  // exact
        if ((fb & 0x7FFFFFFFu) != 0u) {
            const int il = int((fb >> 23) & 0xFFu) - 127;
            fb = ((fb | (1u << 23)) & 0xFFFFFFu) >> uint(metal::min(-il, 24));
        }
        red = sgn | (uint(metal::abs(ip)) << 23) | (fb & 0x7FFFFFu);
    }
    const uint rsign = red >> 31;
    int integral = int((red >> 23) & 0xFFu);
    if ((integral & 0x81) == 0x80)
        return wp_cuda_f(0x7FFFFFFFu);
    if ((integral & 0x81) == 0x81)
        return rsign ? 0.0f : wp_cuda_f(0x7F800000u);
    if (rsign) {
        integral = -integral;
        if (red & 0x7FFFFFu) {
            red = ~red;
            integral -= 1;
        }
    }
    const uint2 c = wp_mufu_ex2_packed[(red >> 17) & 0x3Fu];
    const ulong c0 = c.x, c1 = c.y & 0x1FFFFu, c2 = c.y >> 17;
    const ulong xl = red & 0x1FFFFu;
    ulong s = (c0 << 32) + ((c1 * xl) << 19) + c2 * wp_mufu_square_approx(xl);
    s += 0x6FC4000000000000ul >> 30;
    const float r = wp_cuda_f((127u << 23) | (uint(s >> 34) & 0x7FFFFFu));
    float out;
    if (integral > 127)
        out = wp_cuda_f(0x7F800000u);
    else if (integral < -126)
        out = 0.0f;
    else
        out = wp_cuda_f(uint(integral + 127) << 23) * r;  // exact
    if (square)
        out = out * out;  // a subnormal square flushes, as .ftz does
    return out;
}

// __nv_logf
inline float wp_cuda_logf(float a)
{
    uint ix = wp_cuda_u(a);
    float adj = 0.0f;
    if ((ix & 0x7F800000u) == 0u && (ix & 0x7FFFFFu) != 0u) {
        // subnormal: CUDA scales by 2^23 (exact) and takes 23 off the exponent; f 2^-149 2^23 = 2^(msb - 126) 1.m
        const uint f = ix & 0x7FFFFFu;
        const int msb = 31 - int(metal::clz(f));
        ix = (ix & 0x80000000u) | (uint(msb + 1) << 23) | ((f << uint(23 - msb)) & 0x7FFFFFu);
        adj = -23.0f;
    }
    // (negative normal inputs are also scaled on CUDA; their result is NaN either way)
    const float x = wp_cuda_f(ix);
    const int e = (int(ix) - int(0x3F2AAAABu)) & int(0xFF800000u);
    const float m = wp_cuda_f(uint(int(ix) - e));
    const float ef = metal::fma(float(e), wp_cuda_f(0x34000000u), adj);
    const float f = m - 1.0f;
    float p = metal::fma(wp_cuda_f(0xBE055027u), f, wp_cuda_f(0x3E1039F6u));
    p = metal::fma(p, f, wp_cuda_f(0xBDF8CDCCu));
    p = metal::fma(p, f, wp_cuda_f(0x3E0F2955u));
    p = metal::fma(p, f, wp_cuda_f(0xBE2AD8B9u));
    p = metal::fma(p, f, wp_cuda_f(0x3E4CED0Bu));
    p = metal::fma(p, f, wp_cuda_f(0xBE7FFF22u));
    p = metal::fma(p, f, wp_cuda_f(0x3EAAAA78u));
    p = metal::fma(p, f, -0.5f);
    float r = metal::fma(wp_cuda_mul(f, p), f, f);
    r = metal::fma(ef, wp_cuda_f(0x3F317218u), r);
    if (!(ix < 0x7F800000u))
        r = metal::fma(x, wp_cuda_f(0x7F800000u), wp_cuda_f(0x7F800000u));
    if ((ix & 0x7FFFFFFFu) == 0u)
        r = wp_cuda_f(0xFF800000u);
    return r;
}

// __nv_expf
inline float wp_cuda_expf(float x)
{
    const float t = metal::fma(x, wp_cuda_f(0x3BBB989Du), 0.5f);
    const float s = (t > 0.0f) ? ((t < 1.0f) ? t : 1.0f) : 0.0f;  // cvt.sat (NaN -> 0)
    // fma.rm(s, 252, 12582913) = 12582913 + floor(252 s): the exact sum lies in [2^23, 2^24), where floats are
    // the integers
    uint k = 0u;
    const uint sb = wp_cuda_u(s);
    if (sb != 0u) {
        const int sh = 150 - int(sb >> 23);  // s = mant 2^-sh
        const uint prod = ((sb & 0x7FFFFFu) | 0x800000u) * 252u;
        k = sh >= 32 ? 0u : (prod >> uint(sh));
    }
    const float n = float(int(k) - 126);  // (12582913 + k) - 12583039, exact
    float r = metal::fma(x, wp_cuda_f(0x3FB8AA3Bu), -n);
    r = metal::fma(x, wp_cuda_f(0x32A57060u), r);
    const float e = wp_cuda_ex2(r);
    // e 2^(k - 126) (bits of 12582913 + k shifted into the exponent field); CUDA's non-ftz mul.f32 keeps a
    // subnormal product, rounded to nearest even
    const uint eb = wp_cuda_u(e);
    const int be = int((eb >> 23) & 0xFFu) + int(k) - 126;
    if ((eb & 0x7FFFFFFFu) == 0u || (eb & 0x7F800000u) == 0x7F800000u || be >= 1)
        return e * wp_cuda_f((0x4B400001u + k) << 23);
    const uint mnt = (eb & 0x7FFFFFu) | 0x800000u;
    const int sh = 1 - be;
    if (sh > 25)
        return wp_cuda_f(eb & 0x80000000u);
    const uint q = mnt >> uint(sh), rem = mnt & ((1u << uint(sh)) - 1u), hf = 1u << uint(sh - 1);
    return wp_cuda_f((eb & 0x80000000u) | (q + ((rem > hf || (rem == hf && (q & 1u))) ? 1u : 0u)));
}

// __nv_sinhf
inline float wp_cuda_sinhf(float x)
{
    const uint xb = wp_cuda_u(x);
    const float a = metal::abs(x);
    if (!(a >= 1.0f)) {  // |x| < 1 or NaN
        if ((xb & 0x7F800000u) == 0u)
            return x;  // zero or subnormal: x * x underflows to 0 and CUDA returns fma(0, x, x) = x
        const float x2 = wp_cuda_mul(x, x);
        float p = metal::fma(wp_cuda_f(0x363D0ADAu), x2, wp_cuda_f(0x394FFF49u));
        p = metal::fma(p, x2, wp_cuda_f(0x3C08889Au));
        p = metal::fma(p, x2, wp_cuda_f(0x3E2AAAABu));
        return metal::fma(wp_cuda_mul(x2, p), x, x);
    }
    float k = metal::trunc(wp_cuda_mul(a, wp_cuda_f(0x3FB8AA3Bu)));
    if (metal::abs(k) > 126.0f)
        k = wp_cuda_f((wp_cuda_u(k) & 0x80000000u) | 0x42FC0000u);
    float r = metal::fma(k, wp_cuda_f(0xBF317218u), a);
    r = metal::fma(k, wp_cuda_f(0x3102E308u), r);
    const float sc = wp_cuda_f(wp_cuda_u(k + wp_cuda_f(0x4B40007Du)) << 23);  // 12583037 + k -> 2^(k - 2)
    const float e = wp_cuda_ex2(wp_cuda_mul(r, wp_cuda_f(0x3FB8AA3Bu))) * sc;
    // div.approx.f32(0.125, e) = FMUL(0.125, MUFU.RCP(e)) (e >= 0.5: no pre-scale; a subnormal product, for e above
    // 2^123, cannot change 2 e - q)
    const float q = wp_cuda_mul(0.125f, wp_cuda_rcp(e));
    float res = metal::fma(2.0f, e, -q);
    if (a >= 90.0f)
        res = wp_cuda_f(0x7F800000u);
    return wp_cuda_f((xb & 0x80000000u) | wp_cuda_u(res));
}
