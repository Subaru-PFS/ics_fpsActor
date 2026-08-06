"""Designs the integration tests use, pinned.

/data/pfsDesign is a pile that grows and gets cleaned, and a design's age changes the
answer: tweaking one to 'now' re-projects it, and a field a fortnight old lands most of
its targets outside the patrol regions.  A test that picks "a recent design" therefore
gives a different result every week for reasons that have nothing to do with the code.

Each entry says what it is for, because the choice is not interchangeable.
"""

SCIENCE = 0x59011d629a2ccdb6
"""thetaPhiScan_060_030 -- 2351 science targets on cobras, all valid.

The default for a convergence: every cobra is commanded, so the command partition is
exercised at full width and any refusal is a real finding rather than the design's age.
Carries no guide stars, so it needs `noTweak`.
"""

BLACK_DOT = 0x76e136dd6176c37b
"""dotConvergence -- 2351 BLACKSPOT targets.

Exercises the parked path: no science target anywhere, so it is the design that shows
BLACKSPOT-as-targetType is distinct from FiberStatus.BLACKSPOT, and that "sent to a dot"
and "is dark" are different sets.
"""

SKY_FIELD_WITH_GUIDE_STARS = 0x5a5947272b5b3b83
"""sky_field_test_tel7_run29 -- 1180 science targets and 68 guide stars.

The only pinned design with guide stars, so the only one that can exercise
tweakTargetPosition: an empty guideStars array gives a SkyCoord with no differentials and
apply_space_motion raises.  Obstime 2026-07-23, so tweaking it to 'now' invalidates most
of the field -- which is itself the case worth testing, not a defect.
"""

HOME_MASK_FILE = "MOD4_group1"
"""A maskFile under /data/fps/maskFiles, 590 cobras.

Small enough that "which cobras did fps command" is a visible subset rather than
everything, which is what makes CobraCommand.NOT_COMMANDED testable at all.
"""
