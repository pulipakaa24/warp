// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CUDA's --use_fast_math float32 reciprocal, division and square root, bit for bit, for Metal.
//
// NVRTC --use_fast_math (Warp's fast_math module option) compiles x / y to div.approx.ftz.f32, i.e. SASS
// MUFU.RCP + FMUL.FTZ, and sqrtf to sqrt.approx.ftz.f32 = MUFU.SQRT (measured on an L4, sm_89, Warp 1.16: a / b in
// a fast_math kernel equals FMUL.FTZ(a, MUFU.RCP(b)) on 2^24 random pairs and special values). MUFU results come
// from NVIDIA's multifunction interpolator (Oberman & Siu, ARITH-17 2005): a table indexed by the top m significand
// bits holds C0, C1, C2; the result is C0 -/+ C1 xl +/- C2 sq(xl) in fixed point, sq a truncated squarer, plus a
// bias, truncated. The term alignments, biases and tables below are those of Benjamin Carleton's ptx-math /
// pyxis-roc sass-math (MIT License, Copyright (c) 2020 Benjamin Carleton, see licenses/sass-math-LICENSE.txt),
// reimplemented with 32-bit integer arithmetic. They reproduce the L4's MUFU.RCP for every float32 in [1, 2) and
// MUFU.SQRT for every float32 in [1, 4) bit for bit; both are exponent-invariant on the L4 (checked over the whole
// normal range), so they hold for every normal input. Denormal inputs and outputs are flushed to zero (.ftz).
//
// Used by fast_math modules when warp.config.metal_fast_math == "cuda" (builtin.h: WP_FDIV, metal_crt.h: sqrtf).


// The MUFU squarer drops the 19 low columns of its partial-product array: sq(x) = x^2 - err(x),
// err(x) = sum over ordered bit pairs i + j <= 17 of x_i x_j 2^(i+j), plus x_9 2^18 (the antidiagonal bit of
// column 18). Splitting x into a low and a high part, the pairs inside the low part are its square, the pairs inside
// the high part fall above column 17, and the mixed pairs form a triangle {p + q <= k} of two bit vectors. A
// triangle is the low half of a carry-free product: with every bit in its own 4-bit field, each field of the
// product holds one column count (<= 9 < 16), and the fields of columns 0..k are summed back with weights 2^s.

// One bit per nibble: 8-bit u -> sum u_p 16^p.
inline uint wp_mufu_spread4(uint u)
{
    u = (u | (u << 12)) & 0x000F000Fu;
    u = (u | (u << 6)) & 0x03030303u;
    return (u | (u << 3)) & 0x11111111u;
}

// Nibble fields c_0..c_7 -> sum c_s 2^s.
inline uint wp_mufu_fold4(uint w)
{
    w = (w & 0x0F0F0F0Fu) + (((w >> 4) & 0x0F0F0F0Fu) << 1);
    w = (w & 0x00FF00FFu) + (((w >> 8) & 0x00FF00FFu) << 2);
    return (w & 0xFFFFu) + ((w >> 16) << 4);
}

// Coefficient tables (sass-math's, packed): {C0, C1 | C2 << 16} for MUFU.RCP (m = 7), {C0, C1 | C2 << 17} for
// MUFU.SQRT (m = 6, rows 64..127 for odd exponents); one 8-byte constant load per evaluation.
// clang-format off
constant uint2 wp_mufu_rcp_packed[128] = {
    {0x3fffffdu, 0x03f5fffeu}, {0x3f80fdfu, 0x03ddfc0au}, {0x3f03f03u, 0x03c6f82du}, {0x3e88cb2u, 0x03b1f467u},
    {0x3e0f83bu, 0x039bf0b6u}, {0x3d980f5u, 0x0386ed1bu}, {0x3d22635u, 0x0373e995u}, {0x3cae757u, 0x035fe622u},
    {0x3c3c3c1u, 0x034de2c3u}, {0x3bcbadbu, 0x033bdf77u}, {0x3b5cc0eu, 0x0329dc3du}, {0x3aef6c8u, 0x0317d914u},
    {0x3a83a82u, 0x0306d5fdu}, {0x3a196b1u, 0x02f6d2f7u}, {0x39b0ad0u, 0x02e6d001u}, {0x394965fu, 0x02d7cd1bu},
    {0x38e38e2u, 0x02c7ca44u}, {0x387f1dfu, 0x02b8c77cu}, {0x381c0dfu, 0x02aac4c3u}, {0x37ba56fu, 0x029dc218u},
    {0x3759f20u, 0x0290bf7bu}, {0x36fad86u, 0x0282bcebu}, {0x369d034u, 0x0277ba69u}, {0x36406c7u, 0x026ab7f3u},
    {0x35e50d6u, 0x025db589u}, {0x358ae03u, 0x0252b32cu}, {0x3531debu, 0x0246b0dau}, {0x34da033u, 0x023cae94u},
    {0x3483483u, 0x0231ac59u}, {0x342da7eu, 0x0225aa28u}, {0x33d91d2u, 0x021ca803u}, {0x3385a28u, 0x0211a5e7u},
    {0x3333332u, 0x0207a3d6u}, {0x32e1c9eu, 0x01fea1cfu}, {0x329161du, 0x01f59fd1u}, {0x3241f68u, 0x01ea9ddcu},
    {0x31f3830u, 0x01e29bf1u}, {0x31a6031u, 0x01da9a0fu}, {0x3159721u, 0x01d19835u}, {0x310dcbeu, 0x01c99664u},
    {0x30c30c2u, 0x01c1949bu}, {0x30792eeu, 0x01b992dau}, {0x3030302u, 0x01b19121u}, {0x2fe80bfu, 0x01aa8f70u},
    {0x2fa0be7u, 0x01a28dc6u}, {0x2f5a440u, 0x019c8c24u}, {0x2f1498fu, 0x01938a88u}, {0x2ecfb9bu, 0x018d88f4u},
    {0x2e8ba2eu, 0x01868767u}, {0x2e4850eu, 0x017f85e0u}, {0x2e05c09u, 0x01798460u}, {0x2dc3eedu, 0x017382e7u},
    {0x2d82d82u, 0x016e8174u}, {0x2d42799u, 0x01668006u}, {0x2d02d01u, 0x01617e9fu}, {0x2cc3d8bu, 0x015c7d3eu},
    {0x2c8590au, 0x01557be2u}, {0x2c47f4fu, 0x01517a8du}, {0x2c0b02bu, 0x014b793cu}, {0x2bceb75u, 0x014677f1u},
    {0x2b93104u, 0x014076abu}, {0x2b580adu, 0x013c756bu}, {0x2b1da46u, 0x0136742fu}, {0x2ae3da6u, 0x013172f8u},
    {0x2aaaaa9u, 0x012e71c7u}, {0x2a72129u, 0x0129709au}, {0x2a3a0fdu, 0x01236f71u}, {0x2a02a01u, 0x011f6e4du},
    {0x29cbc14u, 0x011b6d2eu}, {0x299570fu, 0x01176c13u}, {0x295fad3u, 0x01126afcu}, {0x292a73cu, 0x010f69eau},
    {0x28f5c28u, 0x010a68dbu}, {0x28c1978u, 0x010767d1u}, {0x288df0bu, 0x010266cau}, {0x285acc3u, 0x010065c8u},
    {0x2828282u, 0x00fb64c9u}, {0x27f6027u, 0x00f863ceu}, {0x27c4597u, 0x00f362d6u}, {0x27932b3u, 0x00f061e2u},
    {0x2762761u, 0x00ed60f2u}, {0x2732385u, 0x00e96005u}, {0x2702702u, 0x00e75f1cu}, {0x26d31beu, 0x00e25e35u},
    {0x26a439eu, 0x00df5d52u}, {0x2675c8au, 0x00de5c73u}, {0x2647c68u, 0x00da5b96u}, {0x261a31fu, 0x00d65abcu},
    {0x25ed097u, 0x00d459e6u}, {0x25c04b8u, 0x00d05912u}, {0x2593f68u, 0x00cd5841u}, {0x2568095u, 0x00cc5774u},
    {0x253c824u, 0x00c756a8u}, {0x2511601u, 0x00c555e0u}, {0x24e6a16u, 0x00c4551bu}, {0x24bc44du, 0x00c15458u},
    {0x2492491u, 0x00bd5397u}, {0x2468aceu, 0x00ba52d9u}, {0x243f6f0u, 0x00b8521eu}, {0x24168e0u, 0x00b65165u},
    {0x23ee08fu, 0x00b450afu}, {0x23c5de6u, 0x00b24ffbu}, {0x239e0d4u, 0x00af4f49u}, {0x2376947u, 0x00ad4e9au},
    {0x234f72bu, 0x00ab4dedu}, {0x2328a6eu, 0x00a94d42u}, {0x2302301u, 0x00a64c99u}, {0x22dc0d0u, 0x00a54bf3u},
    {0x22b63cbu, 0x00a14b4eu}, {0x2290be1u, 0x00a04aacu}, {0x226b901u, 0x009f4a0cu}, {0x2246b1cu, 0x009b496du},
    {0x2222221u, 0x009a48d1u}, {0x21fde01u, 0x00994837u}, {0x21d9eacu, 0x0096479eu}, {0x21b6415u, 0x00954708u},
    {0x2192e29u, 0x00934673u}, {0x216fcdcu, 0x009145e0u}, {0x214d021u, 0x008f454fu}, {0x212a7e5u, 0x008f44c0u},
    {0x2108420u, 0x008c4432u}, {0x20e64c1u, 0x008a43a6u}, {0x20c49bau, 0x0089431cu}, {0x20a32feu, 0x00874293u},
    {0x2082081u, 0x0085420cu}, {0x2061236u, 0x00844187u}, {0x204080eu, 0x00834103u}, {0x2020200u, 0x00824081u}
};
constant uint2 wp_mufu_sqrt_packed[128] = {
    {0x1ffffffu, 0x03f4ffffu}, {0x203fc07u, 0x03dcfe05u}, {0x207f03eu, 0x03c4fc16u}, {0x20bdcd1u, 0x03b0fa33u},
    {0x20fc1ecu, 0x039af85au}, {0x2139fb9u, 0x0388f68cu}, {0x2177660u, 0x0376f4c8u}, {0x21b4609u, 0x0364f30du},
    {0x21f0ed9u, 0x0350f15bu}, {0x222d0f4u, 0x0342efb3u}, {0x2268c7eu, 0x0330ee13u}, {0x22a4199u, 0x031eec7bu},
    {0x22df066u, 0x030eeaebu}, {0x2319903u, 0x0302e964u}, {0x2353b91u, 0x02f0e7e3u}, {0x238d82cu, 0x02e4e66bu},
    {0x23c6ef2u, 0x02d6e4f9u}, {0x23fffffu, 0x02cae38eu}, {0x2438b6du, 0x02bee22au}, {0x2471157u, 0x02b0e0ccu},
    {0x24a91d6u, 0x02a2df74u}, {0x24e0d03u, 0x0298de23u}, {0x25182f6u, 0x028adcd7u}, {0x254f3c6u, 0x027edb91u},
    {0x2585f8au, 0x0276da51u}, {0x25bc658u, 0x026ad916u}, {0x25f2845u, 0x0260d7e0u}, {0x2628565u, 0x0256d6b0u},
    {0x265ddceu, 0x024cd584u}, {0x2693192u, 0x0244d45eu}, {0x26c80c5u, 0x023ad33cu}, {0x26fcb7au, 0x0230d21eu},
    {0x27311c1u, 0x022ad106u}, {0x27653afu, 0x0220cff1u}, {0x2799152u, 0x0218cee1u}, {0x27ccabdu, 0x0212cdd5u},
    {0x27fffffu, 0x020acccdu}, {0x2833129u, 0x01fecbc8u}, {0x2865e49u, 0x01f8cac8u}, {0x289876fu, 0x01f2c9ccu},
    {0x28cacabu, 0x01e8c8d2u}, {0x28fce0au, 0x01e4c7ddu}, {0x292eb9au, 0x01dcc6ebu}, {0x2960569u, 0x01d8c5fdu},
    {0x2991b85u, 0x01cec511u}, {0x29c2dfau, 0x01cac42au}, {0x29f3cd6u, 0x01c4c345u}, {0x2a24826u, 0x01bcc263u},
    {0x2a54ff4u, 0x01bac185u}, {0x2a8544eu, 0x01b2c0a9u}, {0x2ab553fu, 0x01acbfd0u}, {0x2ae52d2u, 0x01a6befau},
    {0x2b14d14u, 0x019ebe26u}, {0x2b4440du, 0x019ebd57u}, {0x2b737cbu, 0x0194bc88u}, {0x2ba2856u, 0x0190bbbdu},
    {0x2bd15b9u, 0x018ebaf5u}, {0x2bfffffu, 0x0184ba2eu}, {0x2c2e731u, 0x0184b96bu}, {0x2c5cb59u, 0x017cb8a9u},
    {0x2c8ac80u, 0x0178b7eau}, {0x2cb8aafu, 0x0176b72eu}, {0x2ce65f1u, 0x0170b673u}, {0x2d13e4du, 0x016cb5bbu},
    {0x2d413ccu, 0x05956a08u}, {0x2d9b657u, 0x0573673cu}, {0x2df4dd3u, 0x05596482u}, {0x2e4da82u, 0x053961d6u},
    {0x2ea5ca1u, 0x051f5f3au}, {0x2efd46bu, 0x05035cacu}, {0x2f54219u, 0x04e55a2cu}, {0x2faa5e3u, 0x04cb57bau},
    {0x2ffffffu, 0x04af5554u}, {0x305509fu, 0x049952fcu}, {0x30a97f5u, 0x048350b0u}, {0x30fd632u, 0x046f4e70u},
    {0x3150b84u, 0x04554c3au}, {0x31a3817u, 0x04414a10u}, {0x31f5c17u, 0x042747f0u}, {0x32477afu, 0x041945dcu},
    {0x3298b07u, 0x040143d0u}, {0x32e9645u, 0x03f341d0u}, {0x3339992u, 0x03e13fd8u}, {0x3389512u, 0x03c93de8u},
    {0x33d88e9u, 0x03b93c02u}, {0x3427538u, 0x03ad3a26u}, {0x3475a25u, 0x03973850u}, {0x34c37cdu, 0x038b3684u},
    {0x3510e52u, 0x037734beu}, {0x355ddd2u, 0x036f3302u}, {0x35aa66cu, 0x035f314cu}, {0x35f683du, 0x03532f9eu},
    {0x3642362u, 0x03432df6u}, {0x368d7f7u, 0x03312c54u}, {0x36d8616u, 0x03232abau}, {0x3722ddau, 0x031f2928u},
    {0x376cf5cu, 0x030f279au}, {0x37b6ab6u, 0x02ff2612u}, {0x37fffffu, 0x02f72492u}, {0x3848f50u, 0x02e92316u},
    {0x38918bfu, 0x02e521a2u}, {0x38d9c63u, 0x02d92032u}, {0x3921a53u, 0x02c91ec6u}, {0x39692a2u, 0x02c31d62u},
    {0x39b0567u, 0x02b91c02u}, {0x39f72b7u, 0x02ab1aa6u}, {0x3a3daa3u, 0x02a11950u}, {0x3a83d41u, 0x029517feu},
    {0x3ac9aa3u, 0x028f16b2u}, {0x3b0f2dcu, 0x0287156au}, {0x3b545fdu, 0x027b1426u}, {0x3b99419u, 0x027912e8u},
    {0x3bddd41u, 0x026b11acu}, {0x3c22186u, 0x02651076u}, {0x3c660f8u, 0x025f0f44u}, {0x3ca9ba9u, 0x02530e14u},
    {0x3ced1a6u, 0x024d0ceau}, {0x3d30301u, 0x024b0bc4u}, {0x3d72fc8u, 0x023f0aa0u}, {0x3db580au, 0x02350980u},
    {0x3df7bd5u, 0x02350866u}, {0x3e39b39u, 0x0227074cu}, {0x3e7b642u, 0x02230638u}, {0x3ebccffu, 0x021b0526u},
    {0x3efdf7du, 0x02150418u}, {0x3f3edc8u, 0x0211030eu}, {0x3f7f7efu, 0x020b0206u}, {0x3fbfdfdu, 0x02070102u}
};
// clang-format on

// MUFU.RCP (rcp.approx.ftz.f32). The fixed-point sum C0 2^30 - C1 xl 2^17 + C2 (sq(2 xl) >> 2) + bias is kept
// in 32-bit halves; the result fraction is bits 32..56 of it (renormalised by one bit below 2^56).
inline float wp_cuda_rcp(float x)
{
    const uint u = as_type<uint>(x);
    const uint sign = u & 0x80000000u;
    const uint ex = (u >> 23) & 0xFFu;
    const uint f = u & 0x7FFFFFu;
    if (ex == 0u)
        return as_type<float>(sign | 0x7F800000u);
    if (ex == 0xFFu)
        return f != 0u ? as_type<float>(0x7FFFFFFFu) : as_type<float>(sign);
    const uint xl = f & 0xFFFFu;
    const uint2 c = wp_mufu_rcp_packed[f >> 16];
    const uint c1 = c.y & 0xFFFFu, c2 = c.y >> 16;
    const uint y0 = xl & 0xFFu, y1 = xl >> 8;
    const uint tri7 = wp_mufu_fold4(wp_mufu_spread4(y0) * wp_mufu_spread4(y1));
    const uint q = ((y0 * y1 - tri7) << 9) + ((y1 * y1 - (y1 & 1u)) << 16);
    const uint t = c1 * xl;
    const uint a_lo = c.x << 30, b_lo = t << 17, c_lo = c2 * q;
    const uint l1 = a_lo - b_lo;
    const uint l2 = l1 + c_lo;
    const uint l3 = l2 + 0xCFCE0000u;
    const uint hi = (c.x >> 2) - (t >> 15) + metal::mulhi(c2, q) - uint(a_lo < b_lo) + uint(l2 < l1) + uint(l3 < l2);
    const bool top = (hi >> 24) != 0u;
    const uint frac = (top ? (hi >> 1) : hi) & 0x7FFFFFu;
    const int rexp = (top ? 127 : 126) - (int(ex) - 127);
    return as_type<float>(rexp >= 1 ? (sign | (uint(rexp) << 23) | frac) : sign);
}

// div.approx.ftz.f32: FMUL.FTZ(a, MUFU.RCP(b)) (Apple GPUs flush float32 denormal results). The quotient is rounded
// on its own: on the L4, Warp's x / z + y / z and x / z - y / z round both quotients (div.approx is one PTX
// instruction; measured), while the Metal compiler contracts a visible product into a following subtraction (even
// under `fp contract(off)` here, and through fma(a, r, -0)). fma(a, r, z) with z the zero carrying the product's
// sign is a * r rounded once, signed zeros included, and the compiler cannot reduce it to a contractible multiply.
inline float wp_cuda_div(float a, float b)
{
    const float r = wp_cuda_rcp(b);
    const float z = as_type<float>((as_type<uint>(a) ^ as_type<uint>(r)) & 0x80000000u);
    return metal::fma(a, r, z);
}

// MUFU.SQRT (sqrt.approx.ftz.f32): C0 2^33 + C1 xl 2^18 - C2 sq(xl) + bias in 32-bit halves, fraction = bits
// 35..57.
inline float wp_cuda_sqrt(float x)
{
    const uint u = as_type<uint>(x);
    const uint sign = u & 0x80000000u;
    const uint ex = (u >> 23) & 0xFFu;
    const uint f = u & 0x7FFFFFu;
    if (ex == 0u)
        return as_type<float>(sign);
    if (ex == 0xFFu)
        return (f != 0u || sign != 0u) ? as_type<float>(0x7FFFFFFFu) : x;
    if (sign != 0u)
        return as_type<float>(0x7FFFFFFFu);
    const int e = int(ex) - 127;
    const uint odd = uint(e) & 1u;
    const int half_e = (e - int(odd)) >> 1;
    const uint2 c = wp_mufu_sqrt_packed[(f >> 17) + (odd << 6)];
    const uint c1 = c.y & 0x1FFFFu, c2 = c.y >> 17;
    const uint xl = f & 0x1FFFFu;
    const uint x0 = xl & 0x1FFu, x1 = xl >> 9;
    const uint s0 = wp_mufu_spread4(x0 & 0xFFu), s1 = wp_mufu_spread4(x1);
    const uint tri8 = wp_mufu_fold4(s0 * s1) + ((metal::mulhi(s0, s1) & 15u) << 8) + ((x0 >> 8) & x1 & 1u) * 256u;
    const uint p = c2 * (x0 * x1 - tri8);       // weight 2^10
    const uint r = c2 * (x1 * x1 - (x1 & 1u));  // weight 2^18
    const uint m_lo = c1 * xl, m_hi = metal::mulhi(c1, xl);  // weight 2^18
    // s = c0 2^33 + m 2^18 - p 2^10 - r 2^18 + 0x3FFFC0000, as (hi, lo) 32-bit halves
    const uint m18_lo = m_lo << 18, m18_hi = (m_hi << 18) | (m_lo >> 14);
    const uint p_lo = p << 10, p_hi = p >> 22;
    const uint r_lo = r << 18, r_hi = r >> 14;
    const uint l1 = m18_lo - p_lo;
    const uint l2 = l1 - r_lo;
    const uint l3 = l2 + 0xFFFC0000u;
    const uint hi = (c.x << 1) + m18_hi - p_hi - r_hi + 3u - uint(m18_lo < p_lo) - uint(l1 < r_lo) + uint(l3 < l2);
    return as_type<float>((uint(127 + half_e) << 23) | ((hi >> 3) & 0x7FFFFFu));
}
