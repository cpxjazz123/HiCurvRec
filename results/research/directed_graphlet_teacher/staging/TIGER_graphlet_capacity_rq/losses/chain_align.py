#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5

            # Capacity side: the near pair of this structure must actually share the first
            # code, so the term pulls it up to NEAR_TARGET instead of only separating it
            # from the far pair.  This is the only place the arm spends capacity.
            if kind == "chain":
                near = (s_ab + s_bc) / 2.0
            elif kind == "fork":
                near = s_bc
            elif kind == "merge":
                near = s_ab
            else:
#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5

#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5

#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5

#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5

#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5

#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5

#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5

#: Capacity arm.  The structural terms above only push far pairs apart, which lowers the
#: collision rate; this term pulls structurally NEAR pairs together and is therefore what
#: raises it.  The target is the L1 sharing a near pair should reach, and the weight keeps
#: the term at half the structural weight so the injected norm stays inside the budget.
NEAR_TARGET = 0.20
ATTR_WEIGHT = 0.5
































































                continue
            # Capacity side: the near pair of this structure must actually share the first
            # code, so the term pulls it up to NEAR_TARGET instead of only separating it
            # from the far pair.  This is the only place the arm spends capacity.
            if kind == "chain":
                near = (s_ab + s_bc) / 2.0
            elif kind == "fork":
                near = s_bc
            elif kind == "merge":
                near = s_ab
            else:
                continue
            # Capacity side: the near pair of this structure must actually share the first
            # code, so the term pulls it up to NEAR_TARGET instead of only separating it
            # from the far pair.  This is the only place the arm spends capacity.
            if kind == "chain":
                near = (s_ab + s_bc) / 2.0
            elif kind == "fork":
                near = s_bc
            elif kind == "merge":
                near = s_ab
            else:
                continue
            # Capacity side: the near pair of this structure must actually share the first
            # code, so the term pulls it up to NEAR_TARGET instead of only separating it
            # from the far pair.  This is the only place the arm spends capacity.
            if kind == "chain":
                near = (s_ab + s_bc) / 2.0
            elif kind == "fork":
                near = s_bc
            elif kind == "merge":
                near = s_ab
            else:
                continue
            # Capacity side: the near pair of this structure must actually share the first
            # code, so the term pulls it up to NEAR_TARGET instead of only separating it
            # from the far pair.  This is the only place the arm spends capacity.
            if kind == "chain":
                near = (s_ab + s_bc) / 2.0
            elif kind == "fork":
                near = s_bc
            elif kind == "merge":
                near = s_ab
            else:
            # Capacity side: the near pair of this structure must actually share the first
            # code, so the term pulls it up to NEAR_TARGET instead of only separating it
            # from the far pair.  This is the only place the arm spends capacity.
            if kind == "chain":
                near = (s_ab + s_bc) / 2.0
            elif kind == "fork":
                near = s_bc
            elif kind == "merge":
                near = s_ab
            else:
                continue
            attraction = torch.clamp(NEAR_TARGET - near, min=0.0)
            hinges[kind + "_near"] = float(attraction.mean().detach())
            parts.append(ATTR_WEIGHT * attraction.mean())