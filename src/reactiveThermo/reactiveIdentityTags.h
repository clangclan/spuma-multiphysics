// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_IDENTITY_TAGS_H
#define REACTIVE_IDENTITY_TAGS_H
// Frozen hash-domain bytes from the persisted checkpoint format. These are
// compatibility data, not display names. Renaming them invalidates restarts.
namespace ReactiveIdentityTags {
inline constexpr char fingerprint[]="pintle-reactive-thermo-v3:";
inline constexpr char physical[]="pintle-physical-v1:";
inline constexpr char casePhysics[]="pintle-case-physics-v1:";
inline constexpr char numerical[]="pintle-numerics-v2.1.1:reference-flash:phase-temperature-restart-v1:exact-PR32:fd-half-5e-3:";
}
#endif
