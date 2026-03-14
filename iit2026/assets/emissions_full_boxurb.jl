### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000001
begin
    using ModelingToolkit
    using ModelingToolkit: t, D, setp
    using EarthSciMLBase, EarthSciData
    using DynamicQuantities
    using Dates: Dates, DateTime, Second, datetime2unix, unix2datetime, Millisecond
	using TimeZones
    using CairoMakie, MetaGraphsNext, GraphMakie
	using GraphMakie.NetworkLayout
    using PlutoUI, PlutoTeachingTools
    import Latexify
    using DataInterpolations
    using OrdinaryDiffEqTsit5
    using SolarPosition
    PlutoUI.TableOfContents()
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000002
@register_unit ppb 1

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000003
md"""
# Session 5: Emissions and the Full BOXURB Model

**References**: 
* Middleton (1998), Section 11
* Mangaraj et al. (2024), *Urban Climate* 55, 101928

---

In Session 4 we implemented the last two physics components of the BOXURB model — dispersion parameter and box height. But our emission rate *Q* was still a fixed constant. In reality, emissions vary strongly with time of day and day of week because different source sectors (transport, industry, residential cooking, power generation) follow different activity schedules.

In this session we build a **Mumbai-specific emissions component** using data from Mangaraj et al. (2024), who developed a dedicated emission inventory for Mumbai. We then couple this to the full physics model from previous sessions and explore policy scenarios interactively.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000004
md"""
## Learning Objectives

After completing this notebook, you will be able to:

- Describe the major NOₓ emission sectors in Mumbai and their relative contributions
- Implement time-varying emission profiles using `DataInterpolations.jl` and `@register_symbolic`
- Build a `ModelingToolkit` emissions component with tracked physical units
- Couple the emissions component to the full BOXURB physics model
- Explore how sector-specific emission controls and climate change affect NO₂ concentrations
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000005
md"""
## 1. Recap: The Full BOXURB Physics Model (Session 4)

In Sessions 1–4 we built the BOXURB model piece by piece:

| Component | Equations | Session |
|:---|:---|:---:|
| Basic box model (advection + emission) | Eq. 1–2 | 1 |
| Sensible heat flux H | Eqs. 3–5 | 2 |
| Cloud correction cₐ | Section 3 | 2 |
| Friction velocity u\*, boundary layer depth zᵢ | Eqs. 6–10 | 3 |
| Stability index P | Eqs. 11–12 | 3 |
| Dispersion σz, box height h | Eqs. 13–24 | 4 |

The **one missing piece** is time-varying emissions (Section 11). Currently our box model uses a fixed emission rate `Q = 0.25 kg/s`. In this session, we replace that constant with a realistic, sector-resolved emission model for Mumbai.

The full physics model from Sessions 2–4 (including dispersion parameter and box height) is loaded in the **Appendix** at the end of this notebook.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000006
md"""
## 2. Mumbai's NOₓ Emission Inventory

Mangaraj et al. (2024) developed the first ultra-fine resolution (~400 m) emission inventory for Mumbai, covering 17 anthropogenic source sectors. The total NOₓ emission is **~176 Gg/yr**.

### Sector breakdown (NOₓ, from Table 2 and Fig. 5b of Mangaraj et al.)

| # | Sector | NOₓ (Gg/yr) | Share (%) | Description |
|:---:|:---|:---:|:---:|:---|
| 1 | **Road Transport** | 99.15 | 56.5 | Vehicles on ~3970 km of roads; HCVs, LCVs, buses, 2W, 3W |
| 2 | **Aviation** | 35.44 | 20.2 | Chhatrapati Shivaji Airport; ~46M passengers, 304k flights/yr |
| 3 | **Industry** | 25.27 | 14.4 | ~1479 registered units in 13 industrial zones |
| 4 | **Thermal Power Plant** | 3.77 | 2.1 | Trombay station (1580 MW, coal + oil + gas) |
| 5 | **Diesel Generators** | 2.76 | 1.6 | ~62,500 BTS sites + commercial backup generators |
| 6 | **Municipal Solid Waste** | 2.67 | 1.5 | ~6500 TPD at 3 dumping grounds (Deonar, Mulund, Kanjurmarg) |
| 7 | **Slum** | 2.30 | 1.3 | ~2400 slum clusters; cooking with wood, coal, kerosene |
| 8 | **Household** | 2.18 | 1.2 | ~4M households; LPG, coal, kerosene, wood, cowdung |
| 9 | **Crematory** | 0.72 | 0.4 | 80 Hindu crematoriums (11 electric, rest wood-based) |
| 10 | **Incense/Mosquito Coils** | 0.65 | 0.4 | Residential + street vendor use of incense, coils, cigarettes |
| 11 | **Street Vendor** | 0.65 | 0.4 | >0.25M vendors; cooking with wood, coal, LPG, kerosene |
| | **Total** | **175.6** | **100** | |

Road transport dominates (>56%), followed by aviation (~20%) and industry (~14%). The remaining sectors are individually small but collectively significant (~10%).
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000060
tip(
    md"""
    **Mumbai's vehicle fleet — key differences from the UK**

    Mumbai's road fleet has a very different composition from the UK fleet for which BOXURB was originally calibrated:

    - **Two-wheelers (motorcycles/scooters)**: ~54% of registered vehicles. Two-stroke engines emit disproportionately high NOₓ per km.
    - **Three-wheelers (auto-rickshaws)**: Mostly CNG since the 2000s Supreme Court mandate.
    - **Buses (BEST, private)**: ~19% of vehicular NOₓ. Older diesel buses are a significant source; newer CNG buses are cleaner.
    - **HCVs and LCVs**: ~20% of vehicular NOₓ. Port-connected traffic on arterials, predominantly diesel.
    - **Personal cars**: ~14% of vehicular NOₓ; increasingly petrol/CNG.

    ~35% of total vehicles are >15 years old and classified as "super-emitters" under BS-IV norms. These contribute ~49% of vehicular PM₁₀ emissions.
    """
)

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000007
md"""
### 👉 Question 1: Sector Contributions

Looking at the table above, if Mumbai were to convert 100% of its road transport fleet to electric vehicles (zero tailpipe NOₓ), what fraction of total NOₓ emissions would remain? Which sectors would then dominate, and what policy measures could target them?
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000008
answer_q1 = """
Enter your answer here.
""";

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000000a
md"""
## 3. Temporal Emission Profiles

Different sectors emit at different times. We model this with **diurnal profiles** (hour of day) and **day-of-week profiles** (weekday vs. weekend).

### Diurnal profiles by sector

| Sector | Diurnal pattern | Rationale |
|:---|:---|:---|
| Road Transport | Strong double peak (07–10, 17–21) | Commuter rush hours (Eq. 40) |
| Aviation | Peaks at 06–10 and 17–22 | Flight schedules, morning/evening departures |
| Industry | Flat 08–20, low overnight | Factory operating hours |
| Thermal Power Plant | Nearly constant, slight daytime increase | Base load + peak demand |
| Diesel Generators | Peaks during power outage hours (10–14, 18–22) | Backup during grid stress |
| Municipal Solid Waste | Peaks 06–18 (daytime burning) | Waste burning at dump yards |
| Slum | Morning (06–09) and evening (18–21) cooking peaks | Cooking schedule |
| Household | Similar to slum but shifted later | Cooking with cleaner fuels |
| Crematory | Daytime only (08–17) | Cremation hours |
| Incense/Mosquito Coils | Evening peak (17–22) | Evening/night use |
| Street Vendor | Broad daytime + evening peak (06–22) | Vendor operating hours |

### Day-of-week profiles

- **Road Transport**: Saturday ×0.5, Sunday ×0.333 (Eq. 40)
- **Aviation**: Saturday ×0.85, Sunday ×0.75 (slightly reduced weekend flights)
- **Industry**: Saturday ×0.70, Sunday ×0.30
- **Power Plant**: Nearly constant (Saturday ×0.95, Sunday ×0.90)
- **Diesel Generators**: Saturday ×0.80, Sunday ×0.60
- **Waste Burning**: Constant (waste doesn't take weekends off)
- **Slum/Household/Street Vendor**: Slightly higher weekends (more cooking at home)
- **Crematory**: Constant
- **Incense/Coils**: Higher weekends (festivals, home time)

!!! warning
	These temporal profiles are AI-generated and not necessarily suitable for research or policy analysis.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000000b
md"""
### Implementing temporal profiles with DataInterpolations.jl

We use `LinearInterpolation` from `DataInterpolations.jl` to create smooth lookup functions for the diurnal profiles. Since `Dates.hour` and `Dates.dayofweek` cannot be traced symbolically by ModelingToolkit, we create wrapper functions and register them with `@register_symbolic`.

This follows the same pattern we used in Session 2 for the cloud correction factor `ca_lookup`.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000000c
begin
    # ── Wrapper functions for Dates operations ────────────────────────────
    # These extract hour-of-day and day-of-week from a Unix timestamp.
    # We need wrappers because Dates.jl functions can't be symbolically traced.

    # IST offset: UTC+5:30
    const _ist_offset = Dates.Hour(5) + Dates.Minute(30)

    """Return hour of day (0–23) in IST as Float64 from Unix timestamp (seconds)."""
    function _hour_of_day(t_unix)
        dt = unix2datetime(t_unix) + _ist_offset
        return Float64(Dates.hour(dt))
    end
    @register_symbolic _hour_of_day(t_unix)
    ModelingToolkit.get_unit(::typeof(_hour_of_day), args) = u"1"

    """Return day of week (1=Monday, ..., 7=Sunday) in IST as Float64 from Unix timestamp."""
    function _day_of_week(t_unix)
        dt = unix2datetime(t_unix) + _ist_offset
        return Float64(Dates.dayofweek(dt))
    end
    @register_symbolic _day_of_week(t_unix)
    ModelingToolkit.get_unit(::typeof(_day_of_week), args) = u"1"
end

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000000d
md"""
Now we define the diurnal emission profiles for each of the 11 sectors. Each profile is a 25-element array (hours 0–24, where hour 24 wraps to hour 0) giving a dimensionless scaling factor, normalized so the daily average ≈ 1.0.

**Timezone note:** The profiles below are defined in **IST (Indian Standard Time)** convention, where hour 0 corresponds to midnight IST. This is the standard convention for local emission inventories, where activity patterns (e.g., "morning rush" at hours 7–9) refer to local clock time. The `_hour_of_day` helper extracts the hour from the Unix timestamp, and the profile comments describe IST activity patterns directly.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000000e
begin
    # Hours 0–24 (hour 24 = hour 0, for periodic interpolation)
    _hours = Float64.(0:24)

    # ── 1. Road Transport (adapted from BOXURB Eq. 40 for Mumbai) ────────
    _transport_diurnal = [
        0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, # 0–6: overnight
        1.667, 1.667, 1.667,                               # 7–9: morning rush
        1.227, 1.227, 1.227, 1.227, 1.227, 1.227,         # 10–15: midday
        1.684, 1.684, 1.684,                               # 16–18: evening rush
        0.931, 0.931, 0.931, 0.931, 0.931,                 # 19–23: evening
        0.275,                                              # 24 (wrap)
    ]

    # ── 2. Aviation (flight schedule peaks) ──────────────────────────────
    _aviation_diurnal = [
        0.211, 0.106, 0.106, 0.106, 0.211, 0.423,   # 0–5: red-eye arrivals, low
        1.269, 1.692, 1.903, 1.692,               # 6–9: morning departure peak
        1.269, 1.057, 0.952, 1.057, 1.163, 1.269,     # 10–15: midday moderate
        1.48, 1.692, 1.797, 1.586,               # 16–19: evening departure peak
        1.269, 0.846, 0.529, 0.317,               # 20–23: late evening decline
        0.211,                                # 24 (wrap)
    ]

    # ── 3. Industry ──────────────────────────────────────────────────────
    _industry_diurnal = [
        0.353, 0.353, 0.353, 0.353, 0.353, 0.353,   # 0–5: low overnight
        0.588, 0.824,                          # 6–7: ramp up
        1.529, 1.647, 1.647, 1.647, 1.529, 1.647, 1.647, 1.647, 1.529, 1.529, # 8–17: operating
        1.294, 1.059,                          # 18–19: ramp down
        0.706, 0.588, 0.471, 0.353,               # 20–23: evening decline
        0.353,                                # 24 (wrap)
    ]

    # ── 4. Thermal Power Plant ───────────────────────────────────────────
    _powerplant_diurnal = [
        0.85, 0.82, 0.8, 0.8, 0.82, 0.85, # 0–5: base load
        0.9, 0.95,                            # 6–7: morning ramp
        1.1, 1.15, 1.18, 1.2, 1.2, 1.18, 1.15, 1.12, 1.1, 1.08, # 8–17
        1.05, 1.0,                            # 18–19: evening
        0.95, 0.92, 0.9, 0.88,               # 20–23: decline
        0.85,                                  # 24 (wrap)
    ]

    # ── 5. Diesel Generators (power outage peaks) ────────────────────────
    _dieselgen_diurnal = [
        0.335, 0.223, 0.223, 0.223, 0.223, 0.335,   # 0–5: low overnight
        0.558, 0.781,                          # 6–7: morning ramp
        1.005, 1.34, 1.674, 1.786, 1.674, 1.563,     # 8–13: midday peak (grid stress)
        1.34, 1.116, 1.005, 1.116,               # 14–17: afternoon
        1.451, 1.674, 1.563, 1.34,               # 18–21: evening peak (grid stress)
        0.893, 0.558,                          # 22–23: late evening
        0.335,                                # 24 (wrap)
    ]

    # ── 6. Municipal Solid Waste Burning ─────────────────────────────────
    _waste_diurnal = [
        0.229, 0.229, 0.229, 0.229, 0.229, 0.343,   # 0–5: low overnight
        0.914, 1.371,                          # 6–7: morning ignition
        1.714, 1.829, 1.943, 1.943, 1.829, 1.943, 1.829, 1.714, 1.6, 1.371, # 8–17: daytime
        0.914, 0.571,                          # 18–19: dying down
        0.343, 0.229, 0.229, 0.229,               # 20–23: overnight
        0.229,                                # 24 (wrap)
    ]

    # ── 7. Slum (cooking peaks) ──────────────────────────────────────────
    _slum_diurnal = [
        0.253, 0.253, 0.253, 0.253, 0.253, 0.379,   # 0–5: low overnight
        1.768, 2.274, 2.274, 1.768,               # 6–9: morning cooking
        0.632, 0.505, 0.758, 0.505, 0.505, 0.632,     # 10–15: midday low
        0.758, 1.011,                          # 16–17: afternoon
        2.021, 2.274, 2.021, 1.516,               # 18–21: evening cooking
        0.758, 0.379,                          # 22–23: late evening
        0.253,                                # 24 (wrap)
    ]

    # ── 8. Household (similar to slum but slightly different timing) ─────
    _household_diurnal = [
        0.261, 0.261, 0.261, 0.261, 0.261, 0.391,   # 0–5: low overnight
        1.565, 2.087, 2.087, 1.565,               # 6–9: morning cooking (slightly less intense)
        0.783, 0.652, 0.913, 0.652, 0.652, 0.652,     # 10–15: midday
        0.783, 1.043,                          # 16–17: afternoon
        1.826, 2.217, 1.957, 1.565,               # 18–21: evening cooking
        0.913, 0.391,                          # 22–23: late evening
        0.261,                                # 24 (wrap)
    ]

    # ── 9. Crematory (daytime only) ──────────────────────────────────────
    _crematory_diurnal = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # 0–5: closed
        0.3, 0.75,                          # 6–7: opening
        1.8, 2.25, 2.7, 2.7, 2.25, 2.7, 2.7, 2.25, 1.8, 1.2, # 8–17: operating
        0.45, 0.15,                          # 18–19: closing
        0.0, 0.0, 0.0, 0.0,               # 20–23: closed
        0.0,                                # 24 (wrap)
    ]

    # ── 10. Incense Sticks / Mosquito Coils / Cigarettes ─────────────────
    _incense_diurnal = [
        0.416, 0.277, 0.277, 0.277, 0.416, 0.694,   # 0–5: low (some mosquito coils overnight)
        1.11, 1.249,                          # 6–7: morning puja
        0.971, 0.694, 0.555, 0.555, 0.555, 0.555, 0.555, 0.694, 0.832, 1.11, # 8–17: daytime
        1.942, 2.497, 2.636, 2.497,               # 18–21: evening puja + mosquito coils
        1.665, 0.971,                          # 22–23: late evening coils
        0.416,                                # 24 (wrap)
    ]

    # ── 11. Street Vendor ────────────────────────────────────────────────
    _streetvendor_diurnal = [
        0.105, 0.105, 0.105, 0.105, 0.105, 0.316,   # 0–5: mostly closed
        0.842, 1.263,                          # 6–7: morning setup
        1.474, 1.579, 1.474, 1.368, 1.579, 1.368, 1.368, 1.474, 1.579, 1.684, # 8–17: daytime
        1.789, 1.579, 1.263, 0.842,               # 18–21: evening peak (dinner vendors)
        0.421, 0.211,                          # 22–23: closing
        0.105,                                # 24 (wrap)
    ]

    # Build all diurnal interpolations
    _transport_itp = LinearInterpolation(
        _transport_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _aviation_itp = LinearInterpolation(
        _aviation_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _industry_itp = LinearInterpolation(
        _industry_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _powerplant_itp = LinearInterpolation(
        _powerplant_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _dieselgen_itp = LinearInterpolation(
        _dieselgen_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _waste_itp = LinearInterpolation(
        _waste_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _slum_itp = LinearInterpolation(
        _slum_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _household_itp = LinearInterpolation(
        _household_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _crematory_itp = LinearInterpolation(
        _crematory_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _incense_itp = LinearInterpolation(
        _incense_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )
    _streetvendor_itp = LinearInterpolation(
        _streetvendor_diurnal, _hours;
        extrapolation = ExtrapolationType.Periodic
    )

    md"Diurnal profiles defined for all 11 sectors ✓"
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000064
begin
    # Verify that each diurnal profile averages to approximately 1.0 over hours 0–23.
    # This matters because the annual emission total is scaled by these factors;
    # a profile that doesn't average to 1.0 would bias the total emissions up or down.
    _all_diurnal = [
        _transport_diurnal, _aviation_diurnal, _industry_diurnal,
        _powerplant_diurnal, _dieselgen_diurnal, _waste_diurnal,
        _slum_diurnal, _household_diurnal, _crematory_diurnal,
        _incense_diurnal, _streetvendor_diurnal,
    ]
    for (i, prof) in enumerate(_all_diurnal)
        m = sum(prof[1:24]) / 24.0
        @assert isapprox(m, 1.0, atol = 0.1) "Diurnal profile $i has mean $m, expected ≈ 1.0"
    end
    md"Diurnal profile normalization check passed (all means ≈ 1.0) ✓"
end

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000000f
begin
    # ── Day-of-week profiles ──────────────────────────────────────────────
    # 8 values: Monday=1 through Sunday=7, plus Monday=8 (wrap)
    _days = Float64.(1:8)

    # Normalize weekly profiles so the 7-day average equals 1.0.
    # This ensures time-integrated emissions match the stated annual totals.
    # Note: _normalize_weekly uses only elements 1:7 from its input; the 8th
    # element in each array below is ignored (it is replaced by a copy of
    # element 1 to create a periodic wrap point for interpolation).
    _normalize_weekly(v) = (w = v[1:7] .* (7.0 / sum(v[1:7])); [w; w[1]])

    _transport_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.333, 1.0])
    _aviation_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.75, 1.0])
    _industry_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 1.0])
    _powerplant_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.9, 1.0])
    _dieselgen_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 1.0])
    _waste_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    _slum_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.15, 1.0])
    _household_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.15, 1.0])
    _crematory_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    _incense_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 1.15, 1.25, 1.0])
    _streetvendor_weekly = _normalize_weekly([1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.05, 1.0])

    _transport_week_itp = LinearInterpolation(
        _transport_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _aviation_week_itp = LinearInterpolation(
        _aviation_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _industry_week_itp = LinearInterpolation(
        _industry_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _powerplant_week_itp = LinearInterpolation(
        _powerplant_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _dieselgen_week_itp = LinearInterpolation(
        _dieselgen_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _waste_week_itp = LinearInterpolation(
        _waste_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _slum_week_itp = LinearInterpolation(
        _slum_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _household_week_itp = LinearInterpolation(
        _household_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _crematory_week_itp = LinearInterpolation(
        _crematory_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _incense_week_itp = LinearInterpolation(
        _incense_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )
    _streetvendor_week_itp = LinearInterpolation(
        _streetvendor_weekly, _days;
        extrapolation = ExtrapolationType.Periodic
    )

    md"Weekly profiles defined for all 11 sectors ✓"
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000010
begin
    # ── Unified temporal factor function for all sectors ──────────────────
    # Instead of 11 nearly-identical functions, we use a single function
    # that takes a sector index (1–11) and looks up the corresponding
    # diurnal and weekly interpolations from arrays.

    # Sector index constants for readability at call sites
    const SECTOR_TRANSPORT = 1
    const SECTOR_AVIATION = 2
    const SECTOR_INDUSTRY = 3
    const SECTOR_POWERPLANT = 4
    const SECTOR_DIESELGEN = 5
    const SECTOR_WASTE = 6
    const SECTOR_SLUM = 7
    const SECTOR_HOUSEHOLD = 8
    const SECTOR_CREMATORY = 9
    const SECTOR_INCENSE = 10
    const SECTOR_STREETVENDOR = 11

    # Lookup arrays indexed by sector ID
    const _diurnal_itps = [
        _transport_itp, _aviation_itp, _industry_itp, _powerplant_itp,
        _dieselgen_itp, _waste_itp, _slum_itp, _household_itp,
        _crematory_itp, _incense_itp, _streetvendor_itp,
    ]
    const _weekly_itps = [
        _transport_week_itp, _aviation_week_itp, _industry_week_itp,
        _powerplant_week_itp, _dieselgen_week_itp, _waste_week_itp,
        _slum_week_itp, _household_week_itp, _crematory_week_itp,
        _incense_week_itp, _streetvendor_week_itp,
    ]

    function _emis_temporal_factor(t_sec, t0::DateTime, sector::Int)
        t_unix = t_sec + datetime2unix(t0)
        return _diurnal_itps[sector](_hour_of_day(t_unix)) * _weekly_itps[sector](_day_of_week(t_unix))
    end
    @register_symbolic _emis_temporal_factor(t_sec, t0::DateTime, sector::Int)
    ModelingToolkit.get_unit(::typeof(_emis_temporal_factor), args) = u"1"

    md"Registered unified symbolic emission temporal factor function for all 11 sectors ✓"
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000011
md"""
Let's visualize the diurnal profiles to see how they differ across sectors.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000012
let
    hours = 0:0.25:24
    fig = Figure(size = (900, 500))
    ax = Axis(
        fig[1, 1], xlabel = "Hour of day (IST)", ylabel = "Emission scaling factor",
        title = "Weekday diurnal profiles by sector", xticks = 0:3:24
    )

    # Major sectors (thick lines)
    lines!(
        ax, collect(hours), [_transport_itp(h) for h in hours],
        color = :steelblue, linewidth = 2.5, label = "Transport (56.5%)"
    )
    lines!(
        ax, collect(hours), [_aviation_itp(h) for h in hours],
        color = :teal, linewidth = 2, label = "Aviation (20.2%)"
    )
    lines!(
        ax, collect(hours), [_industry_itp(h) for h in hours],
        color = :darkorange, linewidth = 2, label = "Industry (14.4%)"
    )

    # Minor sectors (thinner lines)
    lines!(
        ax, collect(hours), [_powerplant_itp(h) for h in hours],
        color = :gray50, linewidth = 1.5, linestyle = :dash, label = "Power plant (2.1%)"
    )
    lines!(
        ax, collect(hours), [_slum_itp(h) for h in hours],
        color = :firebrick, linewidth = 1.5, label = "Slum (1.3%)"
    )
    lines!(
        ax, collect(hours), [_dieselgen_itp(h) for h in hours],
        color = :olive, linewidth = 1.5, linestyle = :dot, label = "Diesel gen. (1.6%)"
    )
    lines!(
        ax, collect(hours), [_waste_itp(h) for h in hours],
        color = :brown, linewidth = 1.5, linestyle = :dashdot, label = "Waste burning (1.5%)"
    )

    axislegend(ax, position = :rt, labelsize = 9)
    fig
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000013
md"""
### 👉 Question 2: Diurnal Profile Interpretation

Looking at the diurnal profiles above:

1. Which sector has the most **pronounced** diurnal variation? Why does this make physical sense?
2. At what hour of day would you expect the **highest total NOₓ emission rate**, considering both the temporal profile and each sector's share of total emissions?
3. If you wanted to reduce peak-hour NOₓ emissions as much as possible with a single policy, which sector and time period would you target?
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000014
answer_q2 = """
Enter your answer here.
""";

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000016
md"""
## 4. Building the Emissions Component

Now we build a `ModelingToolkit` component that:
1. Has a separate **parameter** for each sector's annual-average emission rate (Gg/yr)
2. Applies the appropriate **temporal profile** (diurnal × weekly) to each sector
3. Sums all sectors to produce the **total emission rate Q** (kg/s)

The conversion from Gg/yr to kg/s is: 1 Gg/yr = 10⁶ kg/yr = 10⁶/(365.25×86400) kg/s ≈ 0.03169 kg/s.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000017
md"""
### 📟 Exercise 1: Complete the emissions equations

The emissions component below defines each sector's contribution as:

```
Q_sector(t) = Q_sector_annual × conversion_factor × temporal_factor(t)
```

Most sector equations are pre-filled. **Your task**: Fill in the three `missing` stubs:

1. **`Q_waste`** — municipal solid waste burning (a simple sector with no special pattern)
2. **`Q_transport`** — road transport (the largest sector, with its own temporal factor)
3. **`Q_total`** — the sum of all eleven sector contributions

Each sector equation follows the same pattern: `Q_sector ~ Q_sector_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_CONSTANT)`. Refer to the pre-filled sectors as a template.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000040
begin
	struct MumbaiEmissionsCoupler; sys; end
	@component function MumbaiEmissions(domain::DomainInfo; name=:MumbaiEmissions)
		@constants begin
			t_one = 1.0, [description="1 second", unit=u"s"]
			Gg_per_yr_to_kg_per_s = 1e6 / (365.25 * 86400.0), [
				description="Conversion Gg/yr → kg/s", unit=u"kg/s"]
		end
		@parameters begin
			t0::DateTime = unix2datetime(get_tref(domain)), [tunable=false]
			Q_transport_ann = 99.15, [description="Road transport NOₓ (Gg/yr)"]
			Q_aviation_ann = 35.44, [description="Aviation NOₓ (Gg/yr)"]
			Q_industry_ann = 25.27, [description="Industry NOₓ (Gg/yr)"]
			Q_powerplant_ann = 3.77, [description="Thermal power plant NOₓ (Gg/yr)"]
			Q_dieselgen_ann = 2.76, [description="Diesel generators NOₓ (Gg/yr)"]
			Q_waste_ann = 2.67, [description="Municipal solid waste burning NOₓ (Gg/yr)"]
			Q_slum_ann = 2.30, [description="Slum cooking/heating NOₓ (Gg/yr)"]
			Q_household_ann = 2.18, [description="Household cooking NOₓ (Gg/yr)"]
			Q_crematory_ann = 0.72, [description="Crematory NOₓ (Gg/yr)"]
			Q_incense_ann = 0.65, [description="Incense/mosquito coils/cigarettes NOₓ (Gg/yr)"]
			Q_streetvendor_ann = 0.65, [description="Street vendor cooking NOₓ (Gg/yr)"]
		end
		@variables begin
			Q_transport(t), [description="Transport emission rate", unit=u"kg/s"]
			Q_aviation(t), [description="Aviation emission rate", unit=u"kg/s"]
			Q_industry(t), [description="Industry emission rate", unit=u"kg/s"]
			Q_powerplant(t), [description="Power plant emission rate", unit=u"kg/s"]
			Q_dieselgen(t), [description="Diesel generator emission rate", unit=u"kg/s"]
			Q_waste(t), [description="Waste burning emission rate", unit=u"kg/s"]
			Q_slum(t), [description="Slum emission rate", unit=u"kg/s"]
			Q_household(t), [description="Household emission rate", unit=u"kg/s"]
			Q_crematory(t), [description="Crematory emission rate", unit=u"kg/s"]
			Q_incense(t), [description="Incense/coils emission rate", unit=u"kg/s"]
			Q_streetvendor(t), [description="Street vendor emission rate", unit=u"kg/s"]
			Q_total(t), [description="Total NOₓ emission rate", unit=u"kg/s"]
		end
		t_sec = t / t_one
		eqs = [
			Q_transport ~ missing, # Exercise 1: road transport sector equation
			Q_aviation ~ Q_aviation_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_AVIATION),
			Q_industry ~ Q_industry_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_INDUSTRY),
			Q_powerplant ~ Q_powerplant_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_POWERPLANT),
			Q_dieselgen ~ Q_dieselgen_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_DIESELGEN),
			Q_waste ~ missing, # Exercise 1: waste burning sector equation
			Q_slum ~ Q_slum_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_SLUM),
			Q_household ~ Q_household_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_HOUSEHOLD),
			Q_crematory ~ Q_crematory_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_CREMATORY),
			Q_incense ~ Q_incense_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_INCENSE),
			Q_streetvendor ~ Q_streetvendor_ann * Gg_per_yr_to_kg_per_s * _emis_temporal_factor(t_sec, t0, SECTOR_STREETVENDOR),
			Q_total ~ missing, # Exercise 1: sum all eleven sectors
		]
		System(eqs, t; name,
			metadata=Dict(CoupleType => MumbaiEmissionsCoupler))
	end
end

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000001c
md"""
### Testing the emissions component

Let's run the MumbaiEmissions component standalone over one week and visualize the sector contributions.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000001e
md"""
Notice how:
- **Transport** dominates the total emission profile with clear rush-hour peaks
- **Aviation** provides the second-largest contribution with its own distinct schedule
- **Industry** has a broad daytime plateau
- **Slum and household cooking** create secondary peaks at mealtimes
- The total emission rate shows **strong weekday peaks** and **weekend suppression**
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000001f
md"""
## 5. Coupling Emissions to the Full BOXURB Model

### 📟 Exercise 2: Write the coupling equation

Now we need to connect our `MumbaiEmissions` component to the `BoxModelNO2` component. The box model has a parameter `Q` (total emission rate in kg/s) that we need to replace with the time-varying emission rate `Q_total` from `MumbaiEmissions`.

**Your task**: Complete the `couple2` method below. You need to:
1. Convert the `Q` parameter in BoxModelNO2 to a variable (using `param_to_var`)
2. Connect `bm.Q ~ em.Q_total`
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000023
md"""
### Building and solving the full coupled model

Now that we have the coupling equations connecting emissions to the box model, let's build and solve the complete system with all components from Sessions 1–5.

We couple all components together — the box model, solar radiation, cloud correction, sensible heat flux, boundary layer depth, stability index, emissions, and OpenAQ observations.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000065
md"""
This is what our coupled system looks like now:
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000027
md"""
## 6. Model vs. Observations

Let's compare our full model (with time-varying emissions from 11 sectors) against the OpenAQ NO₂ observations for Mumbai, Nov 15–22, 2021.

**Important: NOₓ → NO₂ conversion.** Combustion sources emit NOₓ primarily as NO, which is rapidly oxidised to NO₂ by reaction with O₃ in the atmosphere (the NO–NO₂–O₃ photostationary state). The box model forecasts total NOₓ (= NO + NO₂), expressed as NO₂-equivalent mass, but the OpenAQ observations measure NO₂ only. Rather than model the full photochemistry, we apply an empirical [NO₂]:[NOₓ] function from Middleton (1998, Section 12, Eq. 46):

$$[\text{NO}_2] = 2.166 - [\text{NO}_x]\left(1.236 - 3.348 A_{10} + 1.933 A_{10}^2 - 0.326 A_{10}^3\right)$$

where ``A_{10} = \log_{10}([\text{NO}_x])`` and concentrations are in ppb. At high [NOₓ], the ratio [NO₂]/[NOₓ] can be as low as ~0.13, meaning most of the emitted NOₓ remains as NO rather than NO₂. Below 9 ppb NOₓ, the ratio is fixed at 0.722 (most NO has been oxidised after dilution).
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000070
md"""
### 📟 Exercise 3: Implement the NOₓ-to-NO₂ conversion (Eq. 46)

Implement the `_nox_to_no2_ppb` function that converts total NOₓ concentration (ppb) to NO₂ concentration (ppb) using the Derwent and Middleton (1996) empirical relationship shown above.

**Your task**: Complete the function below. The polynomial formula applies when 9 ≤ NOₓ ≤ 1141.5 ppb. Handle the edge cases:
- If `nox_ppb ≤ 0`, return `0.0`
- If `nox_ppb < 9`, return `0.722 * nox_ppb` (low-concentration linear regime)
- If `nox_ppb > 1141.5`, return `0.25 * nox_ppb` (high-concentration cap)
- Otherwise, compute ``A_{10} = \log_{10}(\text{nox\_ppb})`` and apply Eq. 46
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000071
function _nox_to_no2_ppb(nox_ppb)
	0 # Exercise 3: implement Eq. 46
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000073
details(
    "Reference implementation (reveal after attempting)", Markdown.parse(
        """
        ```julia
        function _nox_to_no2_ppb(nox_ppb)
            nox_ppb <= 0 && return 0.0
            nox_ppb < 9.0 && return 0.722 * nox_ppb
            nox_ppb > 1141.5 && return 0.25 * nox_ppb
            A10 = log10(nox_ppb)
            return 2.166 - nox_ppb * (1.236 - 3.348*A10 + 1.933*A10^2 - 0.326*A10^3)
        end
        ```
        """
    )
)

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000074
let
    if !@isdefined(_nox_to_no2_ppb)
        not_defined(:_nox_to_no2_ppb)
    else
        try
            # Reference values computed from the correct Eq. 46 implementation:
            # _nox_to_no2_ppb(0.0)    = 0.0       (zero input)
            # _nox_to_no2_ppb(5.0)    = 3.61      (linear regime: 0.722 * 5)
            # _nox_to_no2_ppb(100.0)  = 35.766    (polynomial: A10=2.0)
            # _nox_to_no2_ppb(1500.0) = 375.0     (cap regime: 0.25 * 1500)
            ref_vals = [0.0, 3.61, 35.766, 375.0]
            student_vals = [_nox_to_no2_ppb(x) for x in [0.0, 5.0, 100.0, 1500.0]]
            if all(isapprox.(student_vals, ref_vals))
                correct(md"NOₓ-to-NO₂ conversion works correctly across all regimes!")
            else
                # Find which regime failed
                labels = ["nox=0 (zero)", "nox=5 (linear)", "nox=100 (polynomial)", "nox=1500 (cap)"]
				txt = md"Keep working!"
                for (i, (sv, rv)) in enumerate(zip(student_vals, ref_vals))
                    if !(sv ≈ rv)
                        txt = md"The $(labels[i]) regime doesn't match the expected result. Check your implementation of Equation 46."
                    end
                end
				keep_working(txt)
            end
        catch e
            if e isa ModelingToolkitBase.ValidationError
                keep_working(md"The system contains a unit mismatch. This can happen if your equations are incorrect, or if you haven't yet replaced the placeholder values.")
            else
                rethrow(e)
            end
        end
    end
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000029
md"""
The model now shows **time-varying emission peaks** that correlate with the observed NO₂ diurnal cycle. The combination of:
- **Rush-hour emission peaks** (transport + aviation) → elevated NOₓ input
- **Nighttime boundary layer collapse** → poor ventilation
- **Weekend emission reduction** → lower Sunday/Saturday concentrations

produces the observed pattern of high nighttime/early morning NO₂ that drops during the day as the boundary layer deepens.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000002a
md"""
## 7. Interactive Policy Explorer

Use the sliders below to adjust emissions from each sector independently, as well as the background temperature and wind speed. This lets you explore:

- **EV transition**: Reduce transport emissions
- **Flight reduction / SAF**: Reduce aviation emissions
- **Industrial controls**: Reduce industry emissions
- **Clean energy**: Reduce power plant and diesel generator emissions
- **Waste management**: Reduce open waste burning
- **Clean cooking**: Reduce slum, household, and street vendor emissions
- **Climate change**: Increase background temperature T_background
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000002b
md"""
**Major sectors:**

| Sector | Scale | Default |
|:---|:---|:---:|
| Transport (56.5%) | $(@bind scale_transport PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Aviation (20.2%) | $(@bind scale_aviation PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Industry (14.4%) | $(@bind scale_industry PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |

**Minor sectors:**

| Sector | Scale | Default |
|:---|:---|:---:|
| Power plant (2.1%) | $(@bind scale_powerplant PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Diesel generators (1.6%) | $(@bind scale_dieselgen PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Waste burning (1.5%) | $(@bind scale_waste PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Slum (1.3%) | $(@bind scale_slum PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Household (1.2%) | $(@bind scale_household PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Crematory (0.4%) | $(@bind scale_crematory PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Incense/coils (0.4%) | $(@bind scale_incense PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |
| Street vendor (0.4%) | $(@bind scale_streetvendor PlutoUI.Slider(0.0:0.05:2.0, default=1.0, show_value=true)) | 1.0 |

**Meteorology:**

| Parameter | Control | Default |
|:---|:---|:---:|
| T_background (K) | $(@bind T_bg_slider PlutoUI.Slider(295.0:0.5:310.0, default=300.0, show_value=true)) | 300.0 |
| Wind speed u (m/s) | $(@bind u_slider PlutoUI.Slider(1.0:0.5:10.0, default=3.0, show_value=true)) | 3.0 |
| City dimension x (km) | $(@bind x_slider PlutoUI.Slider(10.0:1.0:50.0, default=20.0, show_value=true)) | 20.0 |

**City dimension `x`:** This is the downwind fetch distance — the length of the city that the wind "sees" as it carries pollutants toward the receptor. Mumbai's geometry is elongated:
- **North–south extent:** ~45–50 km (the long axis, from Colaba to the northern suburbs)
- **East–west extent:** ~10–25 km (narrower, varying along the peninsula)
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000002d
md"""
### 👉 Question 3: Policy Analysis

Using the interactive explorer above, answer the following:

1. **EV transition**: Set transport scale to 0.5 (50% of fleet is EV). By what approximate percentage do peak NO₂ concentrations decrease? Is this enough to meet the WHO 2021 annual guideline of 10 μg m⁻³?

2. **Climate change**: Reset all emission scales to 1.0, then increase T_background from 300 K to 305 K (representing a warmer climate). How does this affect NO₂ concentrations and why? Are there other processes affected by climate change that aren't included in this model?

3. **Combined policy**: What combination of sector-specific emission reductions brings peak NO₂ below the WHO guideline even under calm conditions (u = 2 m/s)? Try to find the **least aggressive** combination that achieves this.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000002e
answer_q3 = """
Enter your answer here.
""";

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000002f
hint(
    md"""
    Use the interactive sliders to explore how different emission reduction scenarios affect concentrations.
    """
)

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000030
md"""
The policy explorer shows us *what* emission reductions achieve — but *who* bears the costs and benefits of each policy choice is equally important.

### 👉 Question 4: Equity and Environmental Justice

Mumbai's emission inventory reveals that **slum cooking** (1.3%), **street vendors** (0.4%), and **household cooking** (1.2%) are small fractions of total NOₓ but disproportionately affect the urban poor — both as emitters and as the most exposed population.

Meanwhile, **aviation** (20.2%) primarily serves middle-class and upper-class travelers, and **transport** (56.5%) is split between commercial vehicles (HCVs/LCVs serving the economy) and personal vehicles.

Discuss the **equity implications** of these policy options:
- A blanket ban on two-stroke vehicles (affects two-wheeler owners, many of whom are lower-middle class)
- Reducing aviation emissions through airport curfews or sustainable aviation fuel mandates
- Industrial emission controls (affects factory workers and production costs)
- Clean cooking fuel subsidies for slum dwellers
- Congestion pricing in central Mumbai
- Banning open waste burning at municipal dump sites

Which approach(es) would most equitably reduce NO₂ while minimizing harm to vulnerable populations?
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000031
answer_q4 = """
Enter your answer here.
""";

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000063
hint(md"Think about who bears the costs and who receives the benefits of different emission sources.")

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000032
md"""
## Summary

In this session we:

- Built a **sector-resolved emissions component** for Mumbai with **11 individual sectors** using data from Mangaraj et al. (2024), each with its own annual emission rate parameter and temporal profile

- Implemented **diurnal and weekly temporal profiles** using `DataInterpolations.jl` and `@register_symbolic` wrappers for time functions that can't be symbolically traced

- **Coupled** the emissions component to the full BOXURB physics model from Sessions 1–4, replacing the fixed emission rate with realistic time-varying emissions

- Explored **policy scenarios** interactively — including EV transitions, aviation reduction, industrial controls, clean cooking programs, waste management, and climate change impacts

**Key take-aways:**
- Road transport dominates Mumbai's NOₓ emissions (56.5%), followed by aviation (20.2%) and industry (14.4%)
- The interaction between **emission timing** and **boundary layer dynamics** determines pollution episodes — rush-hour emissions trapped under a shallow nighttime/early morning boundary layer create the worst episodes
- **Multi-sector approaches** are needed for WHO compliance, but the **three largest sectors** (transport, aviation, industry) account for ~91% and should be the primary targets
- **Equity considerations** are essential — clean cooking subsidies benefit the most vulnerable, while aviation reduction targets the least vulnerable

**Next:** Several simplifications limit our model's accuracy: we use constant wind speed and temperature, and the empirical dispersion formulas were developed for European conditions. In Session 6, we'll replace these constants with real GEOS-FP meteorological data for Mumbai. In Session 7, we'll ask: can we use neural networks to learn corrections to the empirical equations directly from Mumbai observations, improving predictions without deriving new physics from scratch?
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000033
md"""
## Appendix: Components from Sessions 1–4

The following components and coupling equations were developed in previous sessions. They are used throughout this notebook.
"""

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000034
begin
    begin
        domain = DomainInfo(
            DateTime(2021, 11, 15), DateTime(2021, 11, 22);
            latrange = deg2rad(18.9f0):deg2rad(0.1):deg2rad(19.3f0),
            lonrange = deg2rad(72.7f0):deg2rad(0.1):deg2rad(73.1f0),
            levrange = 1:1,
            u_proto = zeros(Float32, 1, 1, 1, 1),
        )

        # Opaque function: computes clear-sky solar flux from time, observer location
        function _sp_solar_flux(t_sec, t0::DateTime, lat, lon)
            obs = Observer(lat, lon, 0.0)
            dt = t0 + Millisecond(round(Int, t_sec * 1.0e3))
            sp = SolarPosition.solar_position(obs, dt)
            elev = sp.elevation
            return elev > 0 ? 952.7 * sind(elev) : 0.0
        end
        @register_symbolic _sp_solar_flux(t_sec, t0::DateTime, lat, lon)
        ModelingToolkit.get_unit(::typeof(_sp_solar_flux), args) = u"W/m^2"

        ca_interp = LinearInterpolation(
            [1.07, 0.95, 0.72, 0.23, 0.05],
            [0.0, 2.0, 4.0, 6.0, 8.0];
            extrapolation = ExtrapolationType.Constant
        )
        ca_lookup(x) = ca_interp(x)
        @register_symbolic ca_lookup(x)
        ModelingToolkit.get_unit(::typeof(ca_lookup), args) = u"1"

        # ── BoxModelNO2 ──────────────────────────────────────────────────────
        struct BoxModelNO2Coupler
            sys
        end
        @component function BoxModelNO2(; name = :BoxModelNO2)
            @constants begin
                R = 8.314, [description = "Gas constant", unit = u"m^3*Pa/K/mol"]
                MW_NO2 = 46.006e-3, [description = "NO2 molecular mass", unit = u"kg/mol"]
            end
            @parameters begin
                P = 101325, [description = "Atmospheric Pressure", unit = u"Pa"]
                T = 300, [description = "Urban temperature", unit = u"K"]
                Q = 0.25, [description = "NOx emission rate (kg/s)", unit = u"kg/s"]
                x = 20_000, [description = "City downwind dimension (m)", unit = u"m"]
                u_wind = 3.0, [description = "Wind speed at 10 m (m/s)", unit = u"m/s"]
                h = 400, [description = "Polluted layer depth / box height (m)", unit = u"m"]
                C_bg = 3.0, [description = "Background NO2 concentration (ppb)", unit = u"ppb"]
                ppb_const = 1.0e9, [description = "ppb per mixing ratio"]
            end
            @variables begin
                q(t), [description = "NOx emission flux (kg/(m^2*s))", unit = u"kg/(m^2*s)"]
                C(t) = 8.0, [description = "NOx concentration (ppb)", unit = u"ppb"]
                χ(t), [
                        description = "conversion from kg/m3 to ppb for NO2",
                        unit = u"ppb * m^3 / kg",
                    ]
            end
            eqs = [
                χ ~ R * T / P / MW_NO2 * ppb_const,
                # Note: We approximate Mumbai's area as x² = 400 km² (for x = 20 km).
                # Mumbai's actual area is roughly 603 km², so this simplification
                # concentrates emissions into a smaller area, which partially
                # compensates for emissions outside the box boundary that would advect in.
                q ~ Q / x^2,
                D(C) ~ q * χ / h + (C_bg - C) * u_wind / x,
            ]
            return System(
                eqs, t; name,
                metadata = Dict(CoupleType => BoxModelNO2Coupler)
            )
        end

        # ── SensibleHeatFlux ─────────────────────────────────────────────────
        struct SensibleHeatFluxCoupler
            sys
        end
        @component function SensibleHeatFlux(; name = :SensibleHeatFlux)
            @constants begin
                alpha_day = 0.4, [
                        description = "Daytime heat flux coefficient (dimensionless)",
                    ]
                a_night = 0.1748, [
                        description = "Nighttime cloud coefficient (dimensionless)",
                    ]
                b_night = 2.36, [
                        description = "Nighttime offset (dimensionless)",
                    ]
                c_night = 0.375, [
                        description = "Nighttime wind coefficient", unit = u"s/m",
                    ]
                H_ref = 1.0, [
                        description = "Reference heat flux for unit conversion", unit = u"W/m^2",
                    ]
                Z_ref = 100.0, [description = "Reference solar flux", unit = u"W/m^2"]
                ρ = 1.229, [description = "Air density", unit = u"kg/m^3"]
                c_p = 1005.0, [
                        description = "Specific heat capacity of air",
                        unit = u"J/kg/K",
                    ]
                t_norm = 3600.0, [
                        description = "Normalization time for heat store accumulation (1 hour)",
                        unit = u"s",
                    ]
                Z_zero = 0.0, [
                        description = "Zero flux for unit-safe ifelse branch", unit = u"W/m^2",
                    ]
            end
            @parameters begin
                Z = 0.0, [description = "Solar energy flux", unit = u"W/m^2"]
                c_A = 1.07, [description = "Cloud correction factor (dimensionless)"]
                u = 3.0, [description = "Wind speed at 10 m", unit = u"m/s"]
                N_m = 0.0, [description = "Modified cloud amount (dimensionless)"]
                h = 400.0, [description = "Mixing height", unit = u"m"]
                T_background = 300.0, [
                        description = "Background temperature",
                        unit = u"K",
                    ]
                x = 20000.0, [description = "City downwind dimension", unit = u"m"]
                ΔQ_A = 6.0, [description = "Anthropogenic heat flux (Eq. 25)", unit = u"W/m^2"]
                f_s = 0.046, [
                        description = "Urban heat store fraction (Eq. 27; 0=rural, 0.023=suburb, 0.046=city)",
                    ]
                τ_hs = 14400.0, [
                        description = "Heat store decay timescale (Eq. 28; 4 hours for city)",
                        unit = u"s",
                    ]
            end
            @variables begin
                H_day(t), [description = "Daytime rural sensible heat flux (Eqs. 3-5)", unit = u"W/m^2"]
                H_night(t), [description = "Nighttime rural sensible heat flux (Eqs. 3-5)", unit = u"W/m^2"]
                H(t), [description = "Rural sensible heat flux (Eqs. 3-5)", unit = u"W/m^2"]
                H_urban(t), [description = "Urban sensible heat flux (Eq. 25)", unit = u"W/m^2"]
                T(t) = 300.0, [description = "Urban temperature", unit = u"K"]
                ΔQ_H(t) = 0.0, [
                        description = "Urban heat store correction (Eq. 28)", unit = u"W/m^2",
                    ]
            end
            eqs = [
                # Rural sensible heat flux (Eqs. 3-5):
                H_day ~ alpha_day * c_A * (Z - Z_ref),
                H_night ~ (a_night * N_m - b_night) * exp(c_night * u) * H_ref,
                H ~ ifelse(Z > Z_ref, H_day, H_night),
                # Urban heat store (Middleton 1998, Eqs. 27-28):
                D(ΔQ_H) ~ f_s * ifelse(Z > Z_ref, Z - Z_ref, Z_zero) / t_norm
                    - ΔQ_H / τ_hs,
                # Urban sensible heat flux (Eq. 25):
                H_urban ~ H + ΔQ_H + ΔQ_A,
                # Temperature uses urban H (represents city conditions)
                D(T) ~ H_urban / (ρ * c_p * h) + (T_background - T) * u / x,
            ]
            return System(
                eqs, t; name,
                metadata = Dict(CoupleType => SensibleHeatFluxCoupler)
            )
        end

        # ── SolarRadiation ───────────────────────────────────────────────────
        struct SolarRadiationCoupler
            sys
        end
        @component function SolarRadiation(domain::DomainInfo; name = :SolarRadiation)
            @constants t_one = 1.0, [description = "1 second", unit = u"s"]
            @parameters begin
                t0::DateTime = unix2datetime(get_tref(domain)), [tunable = false]
                lat = 19.0, [description = "Latitude (degrees)"]
                lon = 72.8, [description = "Longitude (degrees)"]
            end
            @variables begin
                Z(t), [description = "Solar energy flux", unit = u"W/m^2"]
            end
            eqs = [
                Z ~ _sp_solar_flux(t / t_one, t0, lat, lon),
            ]
            return System(
                eqs, t; name,
                metadata = Dict(CoupleType => SolarRadiationCoupler)
            )
        end

        # ── CloudCorrection ──────────────────────────────────────────────────
        struct CloudCorrectionCoupler
            sys
        end
        @component function CloudCorrection(; name = :CloudCorrection)
            @parameters begin
                N1 = 0.0, [description = "Low cloud cover (oktas)"]
                N2 = 0.0, [description = "Mid-level cloud cover (oktas)"]
                N3 = 1.0, [description = "High cloud cover (oktas)"]
            end
            @variables begin
                N_m(t), [description = "Modified cloud amount Nₘ (oktas)"]
                c_A(t), [description = "Cloud correction factor"]
            end
            eqs = [
                N_m ~ ifelse(
                    N1 + N2 + N3 >= 9, 9,
                    ifelse(
                        N1 + N2 + N3 == 0, 0,
                        ifelse(
                            max(N1, N2) >= N3, N1 + N2 + N3,
                            ifelse(
                                N1 + N2 + N3 < 3, N1 + N2 + N3,
                                ifelse(
                                    N1 + N2 + N3 < 4, N1 + N2 + N3 - 1,
                                    N1 + N2 + N3 - 2,
                                )
                            )
                        )
                    )
                )
                c_A ~ ca_lookup(N_m)
            ]
            System(
                eqs, t; name = name,
                metadata = Dict(CoupleType => CloudCorrectionCoupler)
            )
        end

        # ── BoundaryLayerDepth ───────────────────────────────────────────────
        struct BoundaryLayerDepthCoupler
            sys
        end
        @component function BoundaryLayerDepth(; name = :BoundaryLayerDepth)
            @constants begin
                κ = 0.4, [description = "von Kármán constant"]
                Ω = 7.2921e-5, [description = "Earth rotation rate", unit = u"1/s"]
                zn_coeff = 0.25, [description = "Neutral BLD coefficient"]
                c_stable = 21500.0, [description = "Stable BLD coefficient"]
                c_unstable = 1400.0, [description = "Unstable BLD coefficient"]
                ρ = 1.229, [description = "Air density", unit = u"kg/m^3"]
                c_p = 1005.0, [description = "Specific heat of air", unit = u"J/kg/K"]
                ρcp_ref = 1.0, [description = "Reference ρ·cₚ", unit = u"J/(m^3*K)"]
                u_ref = 1.0, [description = "Reference wind speed", unit = u"m/s"]
                H_ref = 1.0, [description = "Reference heat flux", unit = u"W/m^2"]
                z_ref = 1.0, [description = "Reference height", unit = u"m"]
                S_ref = 1.0, [description = "Reference accumulated flux", unit = u"J/m^2"]
                τ_reset = 60.0, [description = "S_i decay timescale", unit = u"s"]
                deg_to_rad = π / 180, [description = "Degree to radian conversion"]
            end
            @parameters begin
                lat = 19.0, [description = "Latitude (degrees)"]
                z0 = 0.1, [description = "Roughness length", unit = u"m"]
                Δz = 9.9, [description = "Height for log profile", unit = u"m"]
                u_wind = 3.0, [description = "Wind speed at 10 m", unit = u"m/s"]
                H = 0.0, [description = "Sensible heat flux", unit = u"W/m^2"]
            end
            @variables begin
                u_star(t), [description = "Friction velocity", unit = u"m/s"]
                f(t), [description = "Coriolis parameter", unit = u"1/s"]
                zn(t), [description = "Neutral boundary layer depth", unit = u"m"]
                zs(t), [description = "Stable boundary layer depth", unit = u"m"]
                zi(t), [description = "Boundary layer depth", unit = u"m"]
                S_i(t) = 0.0, [description = "Accumulated heat flux", unit = u"J/m^2"]
            end
            eqs = [
                u_star ~ κ * u_wind / log(Δz / z0),
                f ~ 2 * Ω * sin(lat * deg_to_rad),
                zn ~ zn_coeff * u_star / f,
                zs ~ c_stable * (u_star / u_ref)^2 / max(abs(H / H_ref), 1.0e-3)^(1 // 2) * z_ref,
                D(S_i) ~ ifelse(H / H_ref > 0, H, -S_i / τ_reset),
                zi ~ ifelse(
                    H / H_ref > 0,
                    sqrt(zn^2 + c_unstable * max(S_i / S_ref, 0) / (ρ * c_p / ρcp_ref) * z_ref^2),
                    (zn^3 * zs^3 / (zn^3 + zs^3))^(1 // 3)
                ),
            ]
            System(
                eqs, t; name,
                metadata = Dict(CoupleType => BoundaryLayerDepthCoupler)
            )
        end

        # ── StabilityIndex ───────────────────────────────────────────────────
        struct StabilityIndexCoupler
            sys
        end
        @component function StabilityIndex(; name = :StabilityIndex)
            @constants begin
                u_cap = 8.0, [description = "Wind speed cap"]
                u_ref = 1.0, [description = "Reference wind speed", unit = u"m/s"]
                H_ref = 1.0, [description = "Reference heat flux", unit = u"W/m^2"]
            end
            @parameters begin
                u = 3.0, [description = "Wind speed", unit = u"m/s"]
                H = 0.0, [description = "Sensible heat flux", unit = u"W/m^2"]
            end
            @variables begin
                P(t), [description = "Pasquill stability index"]
                Us(t), [description = "Capped wind speed", unit = u"m/s"]
            end
            eqs = [
                Us ~ min(u, u_cap * u_ref),
                P ~ ifelse(
                    H / H_ref > 0,
                    7.0 - (2.26 + 0.019 * (Us / u_ref - 5.6)^2) *
                        (0.1 * H / H_ref + 2.0 + 0.4 * (Us / u_ref)^(3 // 2))^(0.28 - 0.004 * (Us / u_ref - 2.0)^2),
                    3.6 + 10.0^(0.08 - 0.025 * (Us / u_ref)^2) *
                        abs(0.1 * H / H_ref + 1.0e-9)^(1.54 - 0.011 * (Us / u_ref)^2)
                ),
            ]
            System(
                eqs, t; name,
                metadata = Dict(CoupleType => StabilityIndexCoupler)
            )
        end

        # ── Session 2 coupling equations ─────────────────────────────────────
        function EarthSciMLBase.couple2(h::SensibleHeatFluxCoupler, s::SolarRadiationCoupler)
            h, s = h.sys, s.sys
            h = param_to_var(h, :Z)
            return ConnectorSystem([h.Z ~ s.Z], h, s)
        end

        function EarthSciMLBase.couple2(h::SensibleHeatFluxCoupler, c::CloudCorrectionCoupler)
            h, c = h.sys, c.sys
            h = param_to_var(h, :c_A)
            h = param_to_var(h, :N_m)
            return ConnectorSystem(
                [
                    h.c_A ~ c.c_A,
                    h.N_m ~ c.N_m,
                ], h, c
            )
        end

        function EarthSciMLBase.couple2(bm::BoxModelNO2Coupler, hf::SensibleHeatFluxCoupler)
            bm, hf = bm.sys, hf.sys
            bm = param_to_var(bm, :T)
            hf = param_to_var(hf, :u)
            hf = param_to_var(hf, :x)
            return ConnectorSystem(
                [
                    bm.T ~ hf.T,
                    hf.u ~ bm.u_wind,
                    hf.x ~ bm.x,
                ], bm, hf
            )
        end

        # ── Session 3 coupling equations ─────────────────────────────────────
        # NOTE: In Session 4, bm.h comes from BoxHeight (via couple2(bm, bh)),
        # not from BLD directly. This coupling only passes u_wind.
        # For the Session 3 comparison model, bm.h ~ bl.zi is added manually.
        function EarthSciMLBase.couple2(bm::BoxModelNO2Coupler, bl::BoundaryLayerDepthCoupler)
            bm, bl = bm.sys, bl.sys
            bl = param_to_var(bl, :u_wind)
            return ConnectorSystem(
                [
                    bl.u_wind ~ bm.u_wind,
                ], bm, bl
            )
        end

        # BoundaryLayerDepth uses RURAL H (Middleton Section 5: "The boundary
        # layer develops over a considerable distance upwind of the city, with
        # the heat flux and surface roughness as for the rural case.")
        # H in SensibleHeatFlux is the rural flux; H_urban includes the urban
        # heat store and anthropogenic corrections.
        function EarthSciMLBase.couple2(bl::BoundaryLayerDepthCoupler, hf::SensibleHeatFluxCoupler)
            bl, hf = bl.sys, hf.sys
            hf = param_to_var(hf, :h)
            bl = param_to_var(bl, :H)
            return ConnectorSystem(
                [
                    hf.h ~ bl.zi,
                    bl.H ~ hf.H,
                ], bl, hf
            )
        end

        # StabilityIndex uses URBAN H (Middleton Section 9: "The urban heat
        # store adds to the sensible heat flux and leads towards more unstable
        # conditions i.e. reduces P.")
        function EarthSciMLBase.couple2(si::StabilityIndexCoupler, hf::SensibleHeatFluxCoupler)
            si, hf = si.sys, hf.sys
            si = param_to_var(si, :H)
            si = param_to_var(si, :u)
            return ConnectorSystem(
                [
                    si.H ~ hf.H_urban,
                    si.u ~ hf.u,
                ], si, hf
            )
        end
    end

    # ── BoxHeight (Session 4) ───────────────────────────────────────────
    struct BoxHeightCoupler
        sys
    end
    @component function BoxHeight(; name = :BoxHeight)
        @constants begin
            α = 1.25, [description = "Box height scaling factor"]
            x_ref = 1.0, [description = "Reference distance", unit = u"m"]
            z_ref = 1.0, [description = "Reference height", unit = u"m"]
        end
        @parameters begin
            x = 20000.0, [description = "City downwind dimension", unit = u"m"]
            z0 = 0.3, [description = "Surface roughness length", unit = u"m"]
            P = 3.6, [description = "Pasquill stability index"]
            zi = 800.0, [description = "Boundary layer depth", unit = u"m"]
        end
        @variables begin
            # Nondimensionalized inputs
            x_nd(t), [description = "Nondimensional downwind distance"]
            z0_nd(t), [description = "Nondimensional roughness length"]
            # Stability coefficients (Eqs. 13-16)
            K_coeff(t), [description = "K coefficient"]
            L_coeff(t), [description = "L exponent"]
            M_coeff(t), [description = "M exponent"]
            N_coeff(t), [description = "N coefficient"]
            # Log-transformed inputs (Eqs. 17-18)
            S0(t), [description = "log10(z0)"]
            X_poly(t), [description = "log10(x) - 3"]
            # Polynomial coefficients (Eqs. 19-22)
            A_poly(t), [description = "A polynomial"]
            B_poly(t), [description = "B polynomial"]
            Cc_poly(t), [description = "Cc polynomial"]
            Dd_poly(t), [description = "Dd polynomial"]
            # Output
            σz(t), [description = "Vertical dispersion parameter", unit = u"m"]
            h(t), [description = "Box height", unit = u"m"]
        end
        eqs = [
            # Nondimensionalize inputs
            x_nd ~ max(x / x_ref, 1.0),
            z0_nd ~ max(z0 / z_ref, 0.001),
            # Stability coefficients (Eqs. 13-16)
            K_coeff ~ 0.165 - 0.017 * P,
            L_coeff ~ 1.0 - 0.032 * P,
            M_coeff ~ 0.77 - 0.022 * P,
            N_coeff ~ 0.386e-3 * P,
            # Log-transformed inputs (Eqs. 17-18)
            S0 ~ log10(z0_nd),
            X_poly ~ log10(x_nd) - 3.0,
            # Polynomial coefficients (Eqs. 19-22)
            A_poly ~ 1.365 - 0.102 * X_poly + 0.022 * X_poly^2 + 0.05 * X_poly^3,
            B_poly ~ 0.45 - 0.2738 * X_poly + 0.0352 * X_poly^2 + 0.0086 * X_poly^3,
            Cc_poly ~ 0.075 - 0.074 * X_poly + 0.019 * X_poly^2 - 0.0005 * X_poly^3,
            Dd_poly ~ 0.005 - 0.0045 * X_poly + 0.0026 * X_poly^2 - 0.0031 * X_poly^3,
            # Dispersion parameter (Eq. 23)
            σz ~ K_coeff * (x_nd^L_coeff) / (1 + N_coeff * (x_nd^M_coeff)) *
                (A_poly + B_poly * S0 + Cc_poly * S0^2 + Dd_poly * S0^3) * z_ref,
            # Box height (Eq. 24)
            h ~ min(α * σz, zi),
        ]
        System(
            eqs, t; name,
            metadata = Dict(CoupleType => BoxHeightCoupler)
        )
    end

    # ── Session 4 coupling equations ─────────────────────────────────────
    # Couple BoxHeight ↔ BoxModelNO2: box model gets h from BoxHeight, BoxHeight gets x
    function EarthSciMLBase.couple2(bm::BoxModelNO2Coupler, bh::BoxHeightCoupler)
        bm, bh = bm.sys, bh.sys
        bm = param_to_var(bm, :h)
        bh = param_to_var(bh, :x)
        return ConnectorSystem(
            [
                bm.h ~ bh.h,
                bh.x ~ bm.x,
            ], bm, bh
        )
    end

    # Couple BoxHeight ↔ BoundaryLayerDepth: BoxHeight gets zi
    function EarthSciMLBase.couple2(bh::BoxHeightCoupler, bl::BoundaryLayerDepthCoupler)
        bh, bl = bh.sys, bl.sys
        bh = param_to_var(bh, :zi)
        return ConnectorSystem(
            [
                bh.zi ~ bl.zi,
            ], bh, bl
        )
    end

    # Couple BoxHeight ↔ StabilityIndex: BoxHeight gets P
    function EarthSciMLBase.couple2(bh::BoxHeightCoupler, si::StabilityIndexCoupler)
        bh, si = bh.sys, si.sys
        bh = param_to_var(bh, :P)
        return ConnectorSystem(
            [
                bh.P ~ si.P,
            ], bh, si
        )
    end
end

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000001b
let
    if !@isdefined(MumbaiEmissions)
        not_defined(:MumbaiEmissions)
    else
        try
            me = MumbaiEmissions(domain)
            sys = mtkcompile(me)
            prob = ODEProblem(sys, [], (0.0, 86400.0))
            sol = solve(prob, Tsit5(); saveat = 3600.0)
            Q_vals = sol[sys.Q_total]
            if any(ismissing, Q_vals)
                still_missing()
            else
                # Compute reference: total annual-average Q (kg/s) at temporal factor = 1
                Gg_to_kgs = 1.0e6 / (365.25 * 86400.0)
                ann_rates = [99.15, 35.44, 25.27, 3.77, 2.76, 2.67, 2.3, 2.18, 0.72, 0.65, 0.65]
                ref_mean_Q = 6.39668 # sum(ann_rates) * Gg_to_kgs
                # The 24-hour average of Q_total should ≈ ref_mean_Q (temporal factors average to ~1)
                student_mean_Q = sum(Q_vals[1:(end - 1)]) / (length(Q_vals) - 1)
                if isapprox(student_mean_Q, ref_mean_Q, rtol=1e-6)  # default rtol=√eps
                    qmax = round(maximum(Q_vals), digits = 3)
                    qmin = round(minimum(Q_vals), digits = 3)
                    correct(md"MumbaiEmissions works! Q total ranges from $(qmin) to $(qmax) kg/s over 24 hours.")
                else
					@info student_mean_Q, ref_mean_Q, student_mean_Q ≈ ref_mean_Q
                    keep_working(md"Q total values don't match expected reference. Check that you're computing all eleven sectors correctly and summing them.")
                end
            end
        catch e
            if e isa ModelingToolkitBase.ValidationError
                keep_working(md"The system contains a unit mismatch. This can happen if your equations are incorrect, or if you haven't yet replaced the placeholder values.")
            else
                rethrow(e)
            end
        end
    end
end

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000001d
let
    me = MumbaiEmissions(domain)
    sys = mtkcompile(me)
    prob = ODEProblem(sys, [], (0.0, 604800.0))
    sol = solve(prob, Tsit5(); saveat = 1800.0)

    hours_since_start = sol.t ./ 3600

    fig = Figure(size = (900, 550))

    # Top panel: major sectors
    ax1 = Axis(
        fig[1, 1], ylabel = "NOₓ emission rate (kg/s)",
        title = "Mumbai NOₓ emissions by sector — one week",
        xticklabelsvisible = false
    )

    lines!(ax1, hours_since_start, sol[sys.Q_transport], color = :steelblue, linewidth = 2, label = "Transport (56.5%)")
    lines!(ax1, hours_since_start, sol[sys.Q_aviation], color = :teal, linewidth = 1.8, label = "Aviation (20.2%)")
    lines!(ax1, hours_since_start, sol[sys.Q_industry], color = :darkorange, linewidth = 1.8, label = "Industry (14.4%)")
    lines!(ax1, hours_since_start, sol[sys.Q_total], color = :black, linewidth = 2.5, label = "Total")
    axislegend(ax1, position = :rt, labelsize = 9)

    # Bottom panel: minor sectors
    ax2 = Axis(fig[2, 1], xlabel = "Hours since start", ylabel = "NOₓ (kg/s)")
    lines!(ax2, hours_since_start, sol[sys.Q_powerplant], color = :gray50, linewidth = 1.5, label = "Power plant")
    lines!(ax2, hours_since_start, sol[sys.Q_dieselgen], color = :olive, linewidth = 1.5, label = "Diesel gen.")
    lines!(ax2, hours_since_start, sol[sys.Q_waste], color = :brown, linewidth = 1.5, label = "Waste burning")
    lines!(ax2, hours_since_start, sol[sys.Q_slum], color = :firebrick, linewidth = 1.5, label = "Slum")
    lines!(ax2, hours_since_start, sol[sys.Q_household], color = :salmon, linewidth = 1.5, label = "Household")
    lines!(ax2, hours_since_start, sol[sys.Q_crematory], color = :purple, linewidth = 1, label = "Crematory")
    lines!(ax2, hours_since_start, sol[sys.Q_incense], color = :pink, linewidth = 1, label = "Incense/coils")
    lines!(ax2, hours_since_start, sol[sys.Q_streetvendor], color = :darkgreen, linewidth = 1, label = "Street vendor")
    axislegend(ax2, position = :rt, labelsize = 8)

    for d in 0:6
        vlines!(ax1, [d * 24.0], color = (:black, 0.1))
        vlines!(ax2, [d * 24.0], color = (:black, 0.1))
    end
    linkxaxes!(ax1, ax2)
    fig
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000041
begin
	function EarthSciMLBase.couple2(bm::BoxModelNO2Coupler, em::MumbaiEmissionsCoupler)
		bm, em = bm.sys, em.sys
		bm = param_to_var(bm, :Q)
		ConnectorSystem([
			missing, # Exercise 2: connect bm.Q to em.Q_total
		], bm, em)
	end
	execution_tracker=0 # Please don't delete this :)
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000026
let
	execution_tracker
    if !@isdefined(MumbaiEmissions) || !@isdefined(MumbaiEmissionsCoupler)
        not_defined(:MumbaiEmissions)
    else
        try
            # Couple and solve with student's coupling function
            student_coupled = couple(BoxModelNO2(), MumbaiEmissions(domain))
            student_sys = convert(System, student_coupled)
            student_prob = ODEProblem(student_sys, [], (0.0, 86400.0))
            student_sol = solve(student_prob, Tsit5(); saveat = 3600.0)
            C_vals = student_sol[student_sys.BoxModelNO2₊C]
            Q_vals = student_sol[student_sys.MumbaiEmissions₊Q_total]
            # Reference: the annual-average Q should produce a specific steady-state C.
            # At t=24h on a weekday, with temporal modulation, C should settle near
            # a value determined by the emission-advection balance.
            # Check that Q_total daily mean matches expected value from annual totals.
            Gg_to_kgs = 1.0e6 / (365.25 * 86400.0)
            ref_mean_Q = 6.39668 #sum([99.15, 35.44, 25.27, 3.77, 2.76, 2.67, 2.3, 2.18, 0.72, 0.65, 0.65]) * Gg_to_kgs
            student_mean_Q = sum(Q_vals[1:(end - 1)]) / (length(Q_vals) - 1)
            # Compute reference C with a standalone ODE that uses the correct
            # emission coupling (Q = Q_total from MumbaiEmissions), bypassing
            # the student's couple2 method.
            ref_C_final = 47.4554 #let
            #     _R = 8.314; _MW = 46.006e-3; _P = 101325.0; _T = 300.0
            #     _x = 20000.0; _u = 3.0; _h = 400.0; _Cbg = 3.0
            #     _χ = _R * _T / _P / _MW * 1.0e9
            #     _Gg2kgs = 1.0e6 / (365.25 * 86400.0)
            #     _ann = [99.15, 35.44, 25.27, 3.77, 2.76, 2.67, 2.3, 2.18, 0.72, 0.65, 0.65]
            #     _t0 = get_tspan_datetime(domain)[1]
            #     function _Q_ref(t_sec)
            #         return sum(s -> _ann[s] * _Gg2kgs * _emis_temporal_factor(t_sec, _t0, s), 1:11)
            #     end
            #     function _dCdt(C, p, t_sec)
            #         q = _Q_ref(t_sec) / _x^2
            #         return q * _χ / _h + (_Cbg - C) * _u / _x
            #     end
            #     ref_prob = ODEProblem(_dCdt, 8.0, (0.0, 86400.0))
            #     ref_sol = solve(ref_prob, Tsit5(); saveat = 3600.0)
            #     ref_sol.u[end]
            # end
            student_C_final = C_vals[end]
            if !isapprox(student_mean_Q, ref_mean_Q, rtol=1e-4)
                keep_working(md"Emissions Q_total values don't match reference. Check that `bm.Q ~ em.Q_total` is connected correctly.")
            elseif isapprox(student_C_final, ref_C_final, rtol=1e-6) && all(isfinite, C_vals)
                correct(md"Coupling works! C ranges $(round(minimum(C_vals), digits=1))–$(round(maximum(C_vals), digits=1)) ppb over 24 hours.")
            else
                keep_working(md"Your concentration value doesn't match the expected reference. Check your coupling equation.")
            end
        catch e
            if e isa ModelingToolkitBase.ValidationError
                keep_working(md"The system contains a unit mismatch. This can happen if your equations are incorrect, or if you haven't yet replaced the placeholder values.")
			elseif e isa ModelingToolkitBase.MethodError && occursin("missing", string(e))
            keep_working()
            else
                rethrow(e)
            end
        end
    end
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000024
# The OpenAQ component (from EarthSciData) downloads hourly NO₂ observations from
# OpenAQ monitoring stations in the model domain, which we use for model evaluation.
full5_coupled = couple(
    BoxModelNO2(),
    SolarRadiation(domain),
    CloudCorrection(),
    SensibleHeatFlux(),
    BoundaryLayerDepth(),
    StabilityIndex(),
    BoxHeight(),
    MumbaiEmissions(domain),
    OpenAQ("no2", domain; api_key = get(ENV, "OPENAQ_API_KEY", "")),
)

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000066
let
	g = graph(full5_coupled)
	f = Figure(backgroundcolor = :transparent)
	ax = Axis(f[1, 1], backgroundcolor = :transparent,
		xautolimitmargin = (0.18, 0.18),
    	yautolimitmargin = (0.18, 0.18),
	)
	p = graphplot!(ax, g; ilabels=labels(g), edge_color=:gray,
				      layout=Shell(),
				  )
	hidedecorations!(ax); hidespines!(ax)
	f
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000025
begin
	execution_tracker
    full5_sys = convert(System, full5_coupled)
    full5_prob = ODEProblem(full5_sys, [], get_tspan(domain))
    full5_sol = solve(full5_prob, Tsit5(), saveat = 3600)
end

# ╔═╡ e4f5a6b7-0005-0005-0005-000000000028
let
	execution_tracker
    ugm3perkgm3 = 1.0e9
    # Convert solver time (seconds from simulation start) to IST DateTime
    t0_utc = ZonedDateTime(get_tspan_datetime(domain)[1], TimeZone("UTC"))
    to_ist(sol) = DateTime.(astimezone.(t0_utc .+ Second.(sol.t), (TimeZone("Asia/Kolkata"),)))

    obs_no2 = full5_sol[full5_sys.OpenAQ₊no2] .* ugm3perkgm3
    obs_no2 = [v != 0.0 ? v : missing for v in obs_no2]

    # Convert NOₓ → NO₂ using Middleton (1998) Eq. 46, then to μg/m³
    nox_ppb = full5_sol[full5_sys.BoxModelNO2₊C]
    no2_ppb = _nox_to_no2_ppb.(nox_ppb)
    χ_vals = full5_sol[full5_sys.BoxModelNO2₊χ]
    m5 = no2_ppb ./ χ_vals .* ugm3perkgm3
    Q_total = full5_sol[full5_sys.MumbaiEmissions₊Q_total]
    h = full5_sol[full5_sys.BoxHeight₊h]

    fig = Figure(size = (900, 700))

    ax1 = Axis(
        fig[1, 1], ylabel = "NO₂ (μg m⁻³)",
        title = "Full BOXURB + 11-Sector Emissions — Mumbai, Nov 15–22 2021",
        xticklabelsvisible = false
    )
    lines!(
        ax1, to_ist(full5_sol), obs_no2, color = :steelblue, linewidth = 2.5,
        label = "OpenAQ observations"
    )
    lines!(
        ax1, to_ist(full5_sol), m5, color = :firebrick, linewidth = 2,
        label = "Session 5 (full model + emissions)"
    )
    hlines!(ax1, [10.0], color = :orange, linestyle = :dash, label = "WHO 2021 annual guideline (10 μg m⁻³)")
    axislegend(ax1, position = :rt, labelsize = 10)

    ax2 = Axis(
        fig[2, 1], ylabel = "Q_total (kg/s)",
        xticklabelsvisible = false
    )
    lines!(ax2, to_ist(full5_sol), Q_total, color = :black, linewidth = 2)

    ax3 = Axis(fig[3, 1], xlabel = "Date (IST)", ylabel = "Box Height (m)")
    lines!(ax3, to_ist(full5_sol), h, color = :firebrick, linewidth = 2)

    linkxaxes!(ax1, ax2, ax3)
    rowsize!(fig.layout, 2, Relative(0.2))
    rowsize!(fig.layout, 3, Relative(0.2))
    fig
end

# ╔═╡ 0b0af5db-880e-4f16-a013-01d9d138c7ec
begin
    solve_emis = let
		coupled = couple(
	        BoxModelNO2(),
	        SolarRadiation(domain),
	        CloudCorrection(),
	        SensibleHeatFlux(),
	        BoundaryLayerDepth(),
	        StabilityIndex(),
	        BoxHeight(),
	        MumbaiEmissions(domain),
	        OpenAQ("no2", domain; api_key = get(ENV, "OPENAQ_API_KEY", "")),
	    )
	    sys = convert(System, coupled)
    	prob = ODEProblem(sys, [], get_tspan(domain))

        setter = setp(
            sys, [
            sys.MumbaiEmissions₊Q_transport_ann,
            sys.MumbaiEmissions₊Q_aviation_ann,
            sys.MumbaiEmissions₊Q_industry_ann,
            sys.MumbaiEmissions₊Q_powerplant_ann,
            sys.MumbaiEmissions₊Q_dieselgen_ann,
            sys.MumbaiEmissions₊Q_waste_ann,
            sys.MumbaiEmissions₊Q_slum_ann,
            sys.MumbaiEmissions₊Q_household_ann,
            sys.MumbaiEmissions₊Q_crematory_ann,
            sys.MumbaiEmissions₊Q_incense_ann,
            sys.MumbaiEmissions₊Q_streetvendor_ann,
            sys.SensibleHeatFlux₊T_background,
            sys.BoxModelNO2₊u_wind,
            sys.BoxModelNO2₊x,
        	]
        )
        (trans, av, ind, power, dieselgen, waste, slum, house, crem, incense, streetvend, T_bk, u_wind, x) -> begin
            setter(prob, (trans, av, ind, power, dieselgen, waste, slum, house, crem, incense, streetvend, T_bk, u_wind, x))
            solve(prob, Tsit5(); saveat = 3600.0)
        end
    end
    nothing
end

# ╔═╡ e4f5a6b7-0005-0005-0005-00000000002c
let
	sol = solve_emis(99.15 * scale_transport,
            35.44 * scale_aviation,
             25.27 * scale_industry,
           3.77 * scale_powerplant,
            2.76 * scale_dieselgen,
            2.67 * scale_waste,
           2.3 * scale_slum,
         2.18 * scale_household,
          0.72 * scale_crematory,
         0.65 * scale_incense,
      0.65 * scale_streetvendor,
      T_bg_slider,
      u_slider,
     x_slider * 1000.0)

    ugm3perkgm3 = 1.0e9
    # Convert solver time (seconds from simulation start) to IST DateTime
    t0_utc = ZonedDateTime(get_tspan_datetime(domain)[1], TimeZone("UTC"))
    to_ist(sol) = DateTime.(astimezone.(t0_utc .+ Second.(sol.t), (TimeZone("Asia/Kolkata"),)))

    obs_no2 = sol[:OpenAQ₊no2] .* ugm3perkgm3
    obs_no2 = [v != 0.0 ? v : missing for v in obs_no2]
    # Convert NOₓ → NO₂ using Middleton (1998) Eq. 46, then to μg/m³
    nox_ppb_explorer = sol[:BoxModelNO2₊C]
    χ_explorer = sol[:BoxModelNO2₊χ]
    model_no2 = _nox_to_no2_ppb.(nox_ppb_explorer) ./ χ_explorer .* ugm3perkgm3

    total_emis_gg = (
        99.15 * scale_transport + 35.44 * scale_aviation + 25.27 * scale_industry +
            3.77 * scale_powerplant + 2.76 * scale_dieselgen + 2.67 * scale_waste +
            2.3 * scale_slum + 2.18 * scale_household + 0.72 * scale_crematory +
            0.65 * scale_incense + 0.65 * scale_streetvendor
    )

    fig = Figure(size = (900, 450))
    ax = Axis(
        fig[1, 1], xlabel = "Date (IST)", ylabel = "NO₂ (μg m⁻³)",
        title = "Policy Explorer — Total NOₓ: $(round(total_emis_gg, digits = 1)) Gg/yr ($(round(total_emis_gg / 175.6 * 100, digits = 0))% of baseline), T_bg=$(T_bg_slider) K, u=$(u_slider) m/s, x=$(x_slider) km"
    )

    lines!(
        ax, to_ist(sol), obs_no2, color = :steelblue, linewidth = 2.5,
        label = "OpenAQ observations"
    )
    lines!(
        ax, to_ist(sol), model_no2, color = :firebrick, linewidth = 2,
        label = "Model prediction"
    )
    hlines!(ax, [10.0], color = :orange, linestyle = :dash, label = "WHO 2021 annual guideline")
    axislegend(ax, position = :rt, labelsize = 10)
    fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataInterpolations = "82cc6244-b520-54b8-b5a6-8a565e85f1d0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
EarthSciData = "a293c155-435f-439d-9c11-a083b6b47337"
EarthSciMLBase = "e53f1632-a13c-4728-9402-0c66d48804b0"
GraphMakie = "1ecd5474-83a3-4783-bb4f-06765db800d2"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
MetaGraphsNext = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
OrdinaryDiffEqTsit5 = "b1df2697-797e-41e3-8120-5422d3b24e4a"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SolarPosition = "5b9d1343-a731-5a90-8730-7bf8d89bf3eb"
TimeZones = "f269a46b-ccf7-5d73-abea-4c690281aa53"

[sources]
EarthSciData = {rev = "main", url = "https://github.com/EarthSciML/EarthSciData.jl"}

[compat]
CairoMakie = "~0.15.9"
DataInterpolations = "~8.9.0"
DynamicQuantities = "~1.11.0"
EarthSciData = "~0.15.5"
EarthSciMLBase = "~0.25.1"
GraphMakie = "~0.6.3"
Latexify = "~0.16.10"
MetaGraphsNext = "~0.7.5"
ModelingToolkit = "~11.14.0"
OrdinaryDiffEqTsit5 = "~1.9.0"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.79"
SolarPosition = "~0.4.2"
TimeZones = "~1.22.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "901daa49384ae867fdaac429ca666250e5cc7ecc"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "856ecd7cebb68e5fc87abecd2326ad59f0f911f3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.43"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "35ea197a51ce46fcd01c4a44befce0578a1aaeca"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "78b3a7a536b4b0a747a0f296ea77091ca0a9f9a3"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.23.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "e0b47732a192dd59b9d079a06d04235e2f833963"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "308d82aa3d83140909590aa5a7824540944f110f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.8"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijections]]
git-tree-sha1 = "a2d308fcd4c2fb90e943cf9cd2fbfa9c32b69733"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.2.2"

[[deps.BipartiteGraphs]]
deps = ["DataStructures", "DocStringExtensions", "Graphs", "PrecompileTools"]
git-tree-sha1 = "3b050c43d6156f7115f37cf206d3fa34d118c61b"
uuid = "caf10ac8-0290-4205-88aa-f15908547e8d"
version = "0.1.7"
weakdeps = ["SparseArrays"]

    [deps.BipartiteGraphs.extensions]
    BipartiteGraphsSparseArraysExt = "SparseArrays"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "0f606a9894e2bcda541ceb82a91a13c5d450ed97"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.9.3"

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "535c80f1c0847a4c967ea945fca21becc9de1522"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.7+0"

[[deps.BracketingNonlinearSolve]]
deps = ["CommonSolve", "ConcreteStructs", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "4999dff8efd76814f6662519b985aeda975a1924"
uuid = "70df07ce-3d50-431d-a3e7-ca6ddb60ac1e"
version = "1.11.0"
weakdeps = ["ChainRulesCore", "ForwardDiff"]

    [deps.BracketingNonlinearSolve.extensions]
    BracketingNonlinearSolveChainRulesCoreExt = ["ChainRulesCore", "ForwardDiff"]
    BracketingNonlinearSolveForwardDiffExt = "ForwardDiff"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CFTime]]
deps = ["Dates", "Printf"]
git-tree-sha1 = "0836c647014903bedccf23ba72b5ebb8c89a7db8"
uuid = "179af706-886a-5703-950a-314cd64e0468"
version = "0.2.5"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "fa072933899aae6dc61dde934febed8254e66c6a"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.9"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a21c5464519504e41e0cbc91f0188e8ca23d7440"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "1a3ad7e16a321667698a19e77362b35a1e94c544"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.1"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonDataModel]]
deps = ["CFTime", "DataStructures", "Dates", "DiskArrays", "Preferences", "Printf", "Statistics"]
git-tree-sha1 = "cd10f8b38725a6458dd971464daa5a751a67e6b0"
uuid = "1fbeeb36-5f17-413c-809b-666fb144f157"
version = "0.4.2"

[[deps.CommonSolve]]
git-tree-sha1 = "78ea4ddbcf9c241827e7035c3a03e2e456711470"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.6"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "3b4be73db165146d8a88e47924f464e55ab053cd"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.7"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConservativeRegridding]]
deps = ["DocStringExtensions", "Extents", "GeoInterface", "GeometryOps", "GeometryOpsCore", "LinearAlgebra", "ProgressMeter", "SortTileRecursiveTree", "SparseArrays"]
git-tree-sha1 = "bb51e642ddfd67cf6cac0344fabd9c3fe36cd9a3"
uuid = "8e50ac2c-eb48-49bc-a402-07c87b949343"
version = "0.1.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "a692f5e257d332de1e554e4566a4e5a8a72de2b2"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.4"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataInterpolations]]
deps = ["EnumX", "FindFirstFunctions", "ForwardDiff", "LinearAlgebra", "PrettyTables", "RecipesBase", "Reexport"]
git-tree-sha1 = "db37d8739c369b9e7212f8e61e37611bda6fa2e1"
uuid = "82cc6244-b520-54b8-b5a6-8a565e85f1d0"
version = "8.9.0"

    [deps.DataInterpolations.extensions]
    DataInterpolationsChainRulesCoreExt = "ChainRulesCore"
    DataInterpolationsMakieExt = "Makie"
    DataInterpolationsOptimExt = "Optim"
    DataInterpolationsRegularizationToolsExt = "RegularizationTools"
    DataInterpolationsSparseConnectivityTracerExt = ["SparseConnectivityTracer", "FillArrays"]
    DataInterpolationsSymbolicsExt = "Symbolics"

    [deps.DataInterpolations.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"
    RegularizationTools = "29dad682-9a27-4bc3-9c72-016788665182"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "BracketingNonlinearSolve", "ConcreteStructs", "DocStringExtensions", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SciMLStructures", "Setfield", "Static", "StaticArraysCore", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "1719cd1b0a12e01775dc6db1577dd6ace1798fee"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.210.1"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseFlexUnitsExt = "FlexUnits"
    DiffEqBaseForwardDiffExt = ["ForwardDiff"]
    DiffEqBaseGTPSAExt = "GTPSA"
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseMooncakeExt = "Mooncake"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseSparseArraysExt = "SparseArrays"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FlexUnits = "76e01b6b-c995-4ce6-8559-91e72a3d4e95"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffEqCallbacks]]
deps = ["ConcreteStructs", "DataStructures", "DiffEqBase", "DifferentiationInterface", "LinearAlgebra", "Markdown", "PrecompileTools", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "f17b863c2d5d496363fe36c8d8535cc6a33c9952"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "4.12.0"

    [deps.DiffEqCallbacks.extensions]
    DiffEqCallbacksFunctorsExt = "Functors"

    [deps.DiffEqCallbacks.weakdeps]
    Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "7ae99144ea44715402c6c882bfef2adbeadbc4ce"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.16"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DiskArrays]]
deps = ["ConstructionBase", "LRUCache", "Mmap", "OffsetArrays"]
git-tree-sha1 = "e5d9ce1b751ddf9bcd9d36b51249dce8ea73cd55"
uuid = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
version = "0.4.19"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "42cd00edaac86f941815fe557c1d01e11913e07c"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.28"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "fbcc7610f6d8348428f722ecbe0e6cfe22e672c6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.123"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "c249d86e97a7e8398ce2068dce4c078a1c3464de"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.16"
weakdeps = ["Makie", "Random"]

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"
    DomainSetsRandomExt = "Random"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Reexport", "Test"]
git-tree-sha1 = "3f50fa86c968fc1a9e006c07b6bc40ccbb1b704d"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.6.4"

[[deps.DynamicQuantities]]
deps = ["DispatchDoctor", "PrecompileTools", "TestItems", "Tricks"]
git-tree-sha1 = "faa92764e98ab0a1991b06c9b5bb3f1b70b4bf45"
uuid = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
version = "1.11.0"

    [deps.DynamicQuantities.extensions]
    DynamicQuantitiesLinearAlgebraExt = "LinearAlgebra"
    DynamicQuantitiesMeasurementsExt = "Measurements"
    DynamicQuantitiesScientificTypesExt = "ScientificTypes"
    DynamicQuantitiesUnitfulExt = "Unitful"

    [deps.DynamicQuantities.weakdeps]
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EarthSciData]]
deps = ["CodecZlib", "ConservativeRegridding", "DataInterpolations", "Dates", "DiffEqCallbacks", "DocStringExtensions", "Downloads", "DynamicQuantities", "EarthSciMLBase", "Interpolations", "JLD2", "JSON3", "Latexify", "ModelingToolkit", "NCDatasets", "ProgressMeter", "Proj", "SciMLBase", "Scratch", "Symbolics", "TestItemRunner", "ZipFile"]
git-tree-sha1 = "9120138018fbd5e0c46310c74eba9b1761d1fd97"
repo-rev = "main"
repo-url = "https://github.com/EarthSciML/EarthSciData.jl"
uuid = "a293c155-435f-439d-9c11-a083b6b47337"
version = "0.15.5"

[[deps.EarthSciMLBase]]
deps = ["Accessors", "ArrayInterface", "Dates", "DiffEqCallbacks", "DocStringExtensions", "DomainSets", "DynamicQuantities", "Graphs", "LinearAlgebra", "MacroTools", "MetaGraphsNext", "ModelingToolkit", "RuntimeGeneratedFunctions", "SciMLBase", "Statistics", "SymbolicIndexingInterface", "Symbolics", "ThreadsX"]
git-tree-sha1 = "91826db4aa12482c0da2427bcae68140e853fa8b"
uuid = "e53f1632-a13c-4728-9402-0c66d48804b0"
version = "0.25.1"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EnzymeCore]]
git-tree-sha1 = "990991b8aa76d17693a98e3a915ac7aa49f08d1a"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.18"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "01ba9d15e9eae375dc1eb9589df76b3572acd3f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "ab1b34570bcdf272899062e1a56285a53ecaae08"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.3.5"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastPower]]
git-tree-sha1 = "862831f78c7a48681a074ecc9aac09f2de563f71"
uuid = "a4df4552-cc26-4903-aec0-212e50a0e84b"
version = "1.3.1"

    [deps.FastPower.extensions]
    FastPowerEnzymeExt = "Enzyme"
    FastPowerForwardDiffExt = "ForwardDiff"
    FastPowerMeasurementsExt = "Measurements"
    FastPowerMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    FastPowerMooncakeExt = "Mooncake"
    FastPowerReverseDiffExt = "ReverseDiff"
    FastPowerTrackerExt = "Tracker"

    [deps.FastPower.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FindFirstFunctions]]
deps = ["PrecompileTools"]
git-tree-sha1 = "27b495de668ccea58de6b06d6d13181396598ea0"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "1.8.0"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "9340ca07ca27093ff68418b7558ca37b05f8aeb1"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.29.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "eef4c86803f47dcb61e9b8790ecaa96956fdd8ae"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.2"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "7528a7956248c723d01a0a9b0447bf254bf4da52"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.5"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "2b0312a0c06b4408773c6dc1829b472ea706f058"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.6.1"
weakdeps = ["GeometryBasics", "Makie", "RecipesBase"]

    [deps.GeoInterface.extensions]
    GeoInterfaceMakieExt = ["Makie", "GeometryBasics"]
    GeoInterfaceRecipesBaseExt = "RecipesBase"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"
weakdeps = ["GeoInterface"]

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

[[deps.GeometryOps]]
deps = ["AbstractTrees", "AdaptivePredicates", "CoordinateTransformations", "DataAPI", "DelaunayTriangulation", "ExactPredicates", "Extents", "GeoFormatTypes", "GeoInterface", "GeometryOpsCore", "LinearAlgebra", "Random", "SortTileRecursiveTree", "StaticArrays", "Statistics", "Tables"]
git-tree-sha1 = "5c9283d033477a18825eb81bffb389959a2894b3"
uuid = "3251bfac-6a57-4b6d-aa61-ac1fef2975ab"
version = "0.1.38"

    [deps.GeometryOps.extensions]
    GeometryOpsDataFramesExt = "DataFrames"
    GeometryOpsFlexiJoinsExt = "FlexiJoins"
    GeometryOpsLibGEOSExt = "LibGEOS"
    GeometryOpsMakieExt = "Makie"
    GeometryOpsProjExt = "Proj"
    GeometryOpsTGGeometryExt = "TGGeometry"

    [deps.GeometryOps.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    FlexiJoins = "e37f2e79-19fa-4eb7-8510-b63b51fe0a37"
    LibGEOS = "a90b1aa1-3769-5649-ba7e-abc5a9d163eb"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Proj = "c94c279d-25a6-4763-9509-64d165bea63e"
    TGGeometry = "d7e755d2-3c95-4bcf-9b3c-79ab1a78647b"

[[deps.GeometryOpsCore]]
deps = ["DataAPI", "GeoInterface", "StableTasks", "Tables"]
git-tree-sha1 = "3148a79daf82235877a8a49b5a6c35b6d8cf162d"
uuid = "05efe853-fabf-41c8-927e-7063c8b9f013"
version = "0.1.10"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.GraphMakie]]
deps = ["DataStructures", "GeometryBasics", "Graphs", "LinearAlgebra", "Makie", "NetworkLayout", "PolynomialRoots", "SimpleTraits", "StaticArrays"]
git-tree-sha1 = "98b32aaa0007dee58428379c921974b510af0654"
uuid = "1ecd5474-83a3-4783-bb4f-06765db800d2"
version = "0.6.3"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Inflate", "LinearAlgebra", "Random", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7eb45fe833a5b7c51cf6d89c5a841d5967e44be3"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.14.0"
weakdeps = ["Distributed", "SharedArrays"]

    [deps.Graphs.extensions]
    GraphsSharedArraysExt = "SharedArrays"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "157e2e5838984449e44af851a52fe374d56b9ada"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.13.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.ImplicitDiscreteSolve]]
deps = ["ConcreteStructs", "DiffEqBase", "NonlinearSolveBase", "NonlinearSolveFirstOrder", "OrdinaryDiffEqCore", "Reexport", "SciMLBase", "SymbolicIndexingInterface"]
git-tree-sha1 = "80e108544e96389133758dd014b503d8df3cedb8"
uuid = "3263718b-31ed-49cf-8a0f-35a466e8af96"
version = "1.8.0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"
weakdeps = ["ForwardDiff", "Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "02b61501dbe6da3b927cc25dacd7ce32390ee970"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.2"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "d966f85b3b7a8e49d034d27a189e9a4874b4391a"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.13"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "8f8ff711442d1f4cfc0d86133e7ee03d62ec9b98"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.3"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JumpProcesses]]
deps = ["ArrayInterface", "DataStructures", "DiffEqBase", "DiffEqCallbacks", "DocStringExtensions", "FunctionWrappers", "Graphs", "LinearAlgebra", "PoissonRandom", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "StaticArrays", "SymbolicIndexingInterface"]
git-tree-sha1 = "310fc6ddd46db8946f988b142fc2821f2f6456ff"
uuid = "ccbc3e58-028d-4f4c-8cd5-9ae44345cda5"
version = "9.23.1"

    [deps.JumpProcesses.extensions]
    JumpProcessesKernelAbstractionsExt = ["Adapt", "KernelAbstractions"]

    [deps.JumpProcesses.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "4260cfc991b8885bf747801fb60dd4503250e478"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.11"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "125d65fe5042faf078383312dd060adf11d90802"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0205286d9eceadc518742860bf23f703779a3d6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.3+0"

[[deps.LineSearch]]
deps = ["ADTypes", "CommonSolve", "ConcreteStructs", "FastClosures", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "SciMLBase", "SciMLJacobianOperators", "StaticArraysCore"]
git-tree-sha1 = "9f7253c0574b4b585c8909232adb890930da980a"
uuid = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
version = "0.1.6"

    [deps.LineSearch.extensions]
    LineSearchLineSearchesExt = "LineSearches"

    [deps.LineSearch.weakdeps]
    LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DocStringExtensions", "EnumX", "GPUArraysCore", "InteractiveUtils", "Krylov", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "OpenBLAS_jll", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "ba64436736405d666e0d22d54ee0e1b04e2e2b02"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "3.64.0"

    [deps.LinearSolve.extensions]
    LinearSolveAMDGPUExt = "AMDGPU"
    LinearSolveAlgebraicMultigridExt = "AlgebraicMultigrid"
    LinearSolveBLISExt = ["blis_jll", "LAPACK_jll"]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = "CUDA"
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveCUSOLVERRFExt = ["CUSOLVERRF", "SparseArrays"]
    LinearSolveCliqueTreesExt = ["CliqueTrees", "SparseArrays"]
    LinearSolveEnzymeExt = ["EnzymeCore", "SparseArrays"]
    LinearSolveFastAlmostBandedMatricesExt = "FastAlmostBandedMatrices"
    LinearSolveFastLapackInterfaceExt = "FastLapackInterface"
    LinearSolveForwardDiffExt = "ForwardDiff"
    LinearSolveGinkgoExt = ["Ginkgo", "SparseArrays"]
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMetalExt = "Metal"
    LinearSolveMooncakeExt = "Mooncake"
    LinearSolvePETScExt = ["PETSc", "SparseArrays"]
    LinearSolveParUExt = ["ParU_jll", "SparseArrays"]
    LinearSolvePardisoExt = ["Pardiso", "SparseArrays"]
    LinearSolveRecursiveFactorizationExt = "RecursiveFactorization"
    LinearSolveSparseArraysExt = "SparseArrays"
    LinearSolveSparspakExt = ["SparseArrays", "Sparspak"]

    [deps.LinearSolve.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    CUSOLVERRF = "a8cc9031-bad2-4722-94f5-40deabb4245c"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    FastLapackInterface = "29a986be-02c6-4525-aec4-84b980013641"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Ginkgo = "4c8bd3c9-ead9-4b5e-a625-08f1338ba0ec"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    LAPACK_jll = "51474c39-65e3-53ba-86ba-03b1b862ec14"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PETSc = "ace2c81b-2b5f-4b1e-a30d-d662738edfe0"
    ParU_jll = "9e0b026c-e8ce-559c-a2c4-6a3d5c955bc9"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    RecursiveFactorization = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Sparspak = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
    blis_jll = "6136c539-28a5-5bf0-87cc-b183200dce32"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "191686b1ac1ea9c89fc52e996ad15d1d241d1e33"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.10.1+0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9341048b9f723f2ae2a72a5269ac2f15f80534dc"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "36c2d142e7d45fb98b5f83925213feb3292ca348"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.5+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "68af66ec16af8b152309310251ecb4fbfe39869f"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.9"
weakdeps = ["DynamicQuantities"]

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "7eb8cdaa6f0e8081616367c10b31b9d9b34bb02a"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.7"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "54e2fdc38130c05b42be423e90da3bade29b74bd"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.4"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "1983a06112db00c54e171f5cff8788c8efbdae85"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.7.5"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "2c140d60d7cb82badf06d8783800d0bcd1a7daa2"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.8.1"

[[deps.ModelingToolkit]]
deps = ["ADTypes", "BipartiteGraphs", "BlockArrays", "Combinatorics", "CommonSolve", "ConstructionBase", "DataStructures", "DiffEqBase", "DifferentiationInterface", "DocStringExtensions", "FillArrays", "FindFirstFunctions", "ForwardDiff", "Graphs", "InteractiveUtils", "Libdl", "LinearAlgebra", "ModelingToolkitBase", "ModelingToolkitTearing", "Moshi", "OffsetArrays", "OrderedCollections", "PreallocationTools", "PrecompileTools", "REPL", "Reexport", "RuntimeGeneratedFunctions", "SCCNonlinearSolve", "SciMLBase", "SciMLPublic", "Serialization", "Setfield", "SimpleNonlinearSolve", "SparseArrays", "StateSelection", "StaticArrays", "SymbolicIndexingInterface", "SymbolicUtils", "Symbolics", "UnPack"]
git-tree-sha1 = "e8526db50e14eb4a59204b3e9f4d279fac2ff6bf"
uuid = "961ee093-0014-501f-94e3-6117800e7a78"
version = "11.14.0"

    [deps.ModelingToolkit.extensions]
    MTKFMIExt = "FMIImport"
    MTKOrdinaryDiffEqBDFExt = "OrdinaryDiffEqBDF"
    MTKOrdinaryDiffEqDefaultExt = "OrdinaryDiffEqDefault"
    MTKOrdinaryDiffEqRosenbrockExt = "OrdinaryDiffEqRosenbrock"

    [deps.ModelingToolkit.weakdeps]
    FMIImport = "9fcbc62e-52a0-44e9-a616-1359a0008194"
    OrdinaryDiffEqBDF = "6ad6398a-0878-4a85-9266-38940aa047c8"
    OrdinaryDiffEqDefault = "50262376-6c5a-4cf5-baba-aaf4f84d72d7"
    OrdinaryDiffEqRosenbrock = "43230ef6-c299-4910-a778-202eb28ce4ce"

[[deps.ModelingToolkitBase]]
deps = ["ADTypes", "AbstractTrees", "ArrayInterface", "BipartiteGraphs", "BlockArrays", "Combinatorics", "CommonSolve", "Compat", "ConstructionBase", "DataStructures", "DiffEqBase", "DiffEqCallbacks", "DiffRules", "DifferentiationInterface", "DocStringExtensions", "DomainSets", "EnumX", "EnzymeCore", "ExprTools", "FillArrays", "FindFirstFunctions", "ForwardDiff", "FunctionWrappers", "FunctionWrappersWrappers", "Graphs", "ImplicitDiscreteSolve", "InteractiveUtils", "JumpProcesses", "Libdl", "LinearAlgebra", "Moshi", "NaNMath", "OffsetArrays", "OrderedCollections", "PreallocationTools", "PrecompileTools", "REPL", "Random", "ReadOnlyDicts", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLBase", "SciMLPublic", "SciMLStructures", "Serialization", "Setfield", "SimpleNonlinearSolve", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "SymbolicUtils", "Symbolics", "UnPack"]
git-tree-sha1 = "2afdd48284eedd53b89741a07deecd3361aa5efc"
uuid = "7771a370-6774-4173-bd38-47e70ca0b839"
version = "1.21.0"

    [deps.ModelingToolkitBase.extensions]
    MTKBifurcationKitExt = "BifurcationKit"
    MTKCasADiDynamicOptExt = "CasADi"
    MTKChainRulesCoreExt = "ChainRulesCore"
    MTKDiffEqNoiseProcessExt = "DiffEqNoiseProcess"
    MTKDynamicQuantitiesExt = "DynamicQuantities"
    MTKInfiniteOptExt = "InfiniteOpt"
    MTKJuliaFormatterExt = "JuliaFormatter"
    MTKLabelledArraysExt = "LabelledArrays"
    MTKLatexifyExt = "Latexify"
    MTKMooncakeExt = "Mooncake"
    MTKPyomoDynamicOptExt = "Pyomo"

    [deps.ModelingToolkitBase.weakdeps]
    BifurcationKit = "0f109fa4-8a5d-4b75-95aa-f515264e7665"
    CasADi = "c49709b8-5c63-11e9-2fb2-69db5844192f"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffEqNoiseProcess = "77a26b50-5914-5dd7-bc55-306e6241c503"
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    InfiniteOpt = "20393b10-9daf-11e9-18c9-8db751c92c57"
    JuliaFormatter = "98e50ef6-434e-11e9-1051-2b60c6c9e899"
    LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Pyomo = "0e8e1daf-01b5-4eba-a626-3897743a3816"

[[deps.ModelingToolkitTearing]]
deps = ["BipartiteGraphs", "CommonSolve", "DocStringExtensions", "Graphs", "LinearAlgebra", "ModelingToolkitBase", "Moshi", "OffsetArrays", "OrderedCollections", "SciMLBase", "Setfield", "SparseArrays", "StateSelection", "SymbolicIndexingInterface", "SymbolicUtils", "Symbolics", "UUIDs"]
git-tree-sha1 = "b3507c8e852a9dbba1e492184faa82e1d69c7c0e"
uuid = "6bb917b9-1269-42b9-9f7c-b0dca72083ab"
version = "1.6.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "53f817d3e84537d84545e0ad749e483412dd6b2a"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.7"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.MultivariatePolynomials]]
deps = ["DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "d38b8653b1cdfac5a7da3b819c0a8d6024f9a18c"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.13"
weakdeps = ["ChainRulesCore"]

    [deps.MultivariatePolynomials.extensions]
    MultivariatePolynomialsChainRulesCoreExt = "ChainRulesCore"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "22df8573f8e7c593ac205455ca088989d0a2c7a0"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.7"

[[deps.NCDatasets]]
deps = ["CFTime", "CommonDataModel", "DataStructures", "Dates", "DiskArrays", "NetCDF_jll", "NetworkOptions", "Printf"]
git-tree-sha1 = "43e840d07d643a71171ade0f6109d911186bbe10"
uuid = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
version = "0.14.12"

    [deps.NCDatasets.extensions]
    NCDatasetsMPIExt = "MPI"

    [deps.NCDatasets.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetCDF_jll]]
deps = ["Artifacts", "Blosc_jll", "Bzip2_jll", "HDF5_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML", "XML2_jll", "Zlib_jll", "Zstd_jll", "libaec_jll", "libzip_jll"]
git-tree-sha1 = "d574803b6055116af212434460adf654ce98e345"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "401.900.300+0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkLayout]]
deps = ["GeometryBasics", "LinearAlgebra", "Random", "Requires", "StaticArrays"]
git-tree-sha1 = "f7466c23a7c5029dc99e8358e7ce5d81a117c364"
uuid = "46757867-2c16-5918-afeb-47bfcb05e46a"
version = "0.4.10"
weakdeps = ["Graphs"]

    [deps.NetworkLayout.extensions]
    NetworkLayoutGraphsExt = "Graphs"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.NonlinearSolveBase]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "CommonSolve", "Compat", "ConcreteStructs", "DifferentiationInterface", "EnzymeCore", "FastClosures", "LinearAlgebra", "LogExpFunctions", "Markdown", "MaybeInplace", "PreallocationTools", "Preferences", "Printf", "RecursiveArrayTools", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "4f595a0977d6e048fa1e3c382b088b950f8c7934"
uuid = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
version = "2.15.0"

    [deps.NonlinearSolveBase.extensions]
    NonlinearSolveBaseBandedMatricesExt = "BandedMatrices"
    NonlinearSolveBaseChainRulesCoreExt = "ChainRulesCore"
    NonlinearSolveBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    NonlinearSolveBaseForwardDiffExt = "ForwardDiff"
    NonlinearSolveBaseLineSearchExt = "LineSearch"
    NonlinearSolveBaseLinearSolveExt = "LinearSolve"
    NonlinearSolveBaseMooncakeExt = "Mooncake"
    NonlinearSolveBaseReverseDiffExt = "ReverseDiff"
    NonlinearSolveBaseSparseArraysExt = "SparseArrays"
    NonlinearSolveBaseSparseMatrixColoringsExt = "SparseMatrixColorings"
    NonlinearSolveBaseTrackerExt = "Tracker"

    [deps.NonlinearSolveBase.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LineSearch = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
    LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.NonlinearSolveFirstOrder]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConcreteStructs", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "LinearSolve", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLJacobianOperators", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "eea7cbe389b168c77df7ff779fb7277019c685c8"
uuid = "5959db7a-ea39-4486-b5fe-2dd0bf03d60d"
version = "2.0.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f2b3b9e52a5eb6a3434c8cca67ad2dde011194f4"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.30+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "df9b7c88c2e7a2e77146223c526bf9e236d5f450"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "2f3d05e419b6125ffe06e55784102e99325bdbe2"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.10+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.OrdinaryDiffEqCore]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "ConcreteStructs", "DataStructures", "DiffEqBase", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FillArrays", "FunctionWrappersWrappers", "InteractiveUtils", "LinearAlgebra", "Logging", "MacroTools", "MuladdMacro", "Polyester", "PrecompileTools", "Preferences", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Static", "StaticArrayInterface", "StaticArraysCore", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "e051c1fb69b1cb1511a00161b97e7a79e0b70687"
uuid = "bbf590c4-e513-4bbe-9b18-05decba2e5d8"
version = "3.17.0"

    [deps.OrdinaryDiffEqCore.extensions]
    OrdinaryDiffEqCoreMooncakeExt = "Mooncake"
    OrdinaryDiffEqCoreSparseArraysExt = "SparseArrays"

    [deps.OrdinaryDiffEqCore.weakdeps]
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.OrdinaryDiffEqTsit5]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "Static", "TruncatedStacktraces"]
git-tree-sha1 = "8be4cba85586cd2efa6c76d1792c548758610901"
uuid = "b1df2697-797e-41e3-8120-5422d3b24e4a"
version = "1.9.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.PROJ_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "SQLite_jll"]
git-tree-sha1 = "af57004c3b686097d563f9c394d7886431a38c75"
uuid = "58948b4f-47e0-5654-a9ad-f609743f8632"
version = "902.700.100+0"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "90b41ced6bacd8c01bd05da8aed35c5458891749"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PoissonRandom]]
deps = ["LogExpFunctions", "Random"]
git-tree-sha1 = "67afbcbe9e184d6729a92a022147ed4cf972ca7b"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.7"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "16bbc30b5ebea91e9ce1671adc03de2832cff552"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.19"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PolynomialRoots]]
git-tree-sha1 = "5f807b5345093487f733e520a1b7395ee9324825"
uuid = "3a141323-8675-5d76-9d11-e1df1406c778"
version = "1.0.0"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "dc8d6bde5005a0eac05ae8faf1eceaaca166cfa4"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.1.2"

    [deps.PreallocationTools.extensions]
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"
    PreallocationToolsSparseConnectivityTracerExt = "SparseConnectivityTracer"

    [deps.PreallocationTools.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "211530a7dc76ab59087f4d4d1fc3f086fbe87594"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.2.3"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.Proj]]
deps = ["CEnum", "CoordinateTransformations", "GeoFormatTypes", "GeoInterface", "NetworkOptions", "PROJ_jll"]
git-tree-sha1 = "61188669db4f5b400173e4ec60da8bcb72d6e749"
uuid = "c94c279d-25a6-4763-9509-64d165bea63e"
version = "1.9.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.ReadOnlyArrays]]
git-tree-sha1 = "e6f7ddf48cf141cb312b078ca21cb2d29d0dc11d"
uuid = "988b38a3-91fc-5605-94a2-ee2116b3bd83"
version = "0.2.0"

[[deps.ReadOnlyDicts]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "711acef70140078d808be9cd33040f510af57f5e"
uuid = "795d4caa-f5a7-4580-b5d8-c01d53451803"
version = "1.0.1"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "18d2a6fd1ea9a8205cadb3a5704f8e51abdd748b"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.48.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStatisticsExt = "Statistics"
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "02d31ad62838181c1a3a5fd23a1ce5914a643601"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.3"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "7257165d5477fd1025f7cb656019dcb6b0512c38"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.17"

[[deps.SCCNonlinearSolve]]
deps = ["CommonSolve", "PrecompileTools", "Reexport", "SciMLBase", "SymbolicIndexingInterface"]
git-tree-sha1 = "b1fed39f2acdd4524c17b50ce6f4406be61b936d"
uuid = "9dfe8606-65a1-4bb3-9748-cb89d1561431"
version = "1.11.0"
weakdeps = ["ChainRulesCore"]

    [deps.SCCNonlinearSolve.extensions]
    SCCNonlinearSolveChainRulesCoreExt = "ChainRulesCore"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SQLite_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll", "dlfcn_win32_jll"]
git-tree-sha1 = "0b5f220f90642566b65ba86549d1ee4118ab2579"
uuid = "76ed43ae-9a5d-5a62-8c75-30186b810ce8"
version = "3.51.2+0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLLogging", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "4675d321bfebe190d22dc4d9de6af7e318d5174a"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.148.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDifferentiationInterfaceExt = "DifferentiationInterface"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReverseDiffExt = "ReverseDiff"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLJacobianOperators]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DifferentiationInterface", "FastClosures", "LinearAlgebra", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "e96d5e96debf7f80a50d0b976a13dea556ccfd3a"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.12"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "Preferences"]
git-tree-sha1 = "0161be062570af4042cf6f69e3d5d0b0555b6927"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "1.9.1"

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

    [deps.SciMLLogging.weakdeps]
    Tracy = "e689c965-62c8-4b79-b2c5-8359227902fd"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "794c760e6aafe9f40dcd7dd30526ea33f0adc8b7"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.15.1"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "607f6867d0b0553e98fc7f725c9f9f13b4d01a32"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "BracketingNonlinearSolve", "CommonSolve", "ConcreteStructs", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "744c3f0fb186ad28376199c1e72ca39d0c614b5d"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "2.11.0"

    [deps.SimpleNonlinearSolve.extensions]
    SimpleNonlinearSolveChainRulesCoreExt = "ChainRulesCore"
    SimpleNonlinearSolveReverseDiffExt = "ReverseDiff"
    SimpleNonlinearSolveTrackerExt = "Tracker"

    [deps.SimpleNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SolarPosition]]
deps = ["Dates", "DocStringExtensions", "Reexport", "StructArrays", "Tables", "TimeZones"]
git-tree-sha1 = "b0e5878adacbcea4b65e5d40ce2d734e90e59079"
uuid = "5b9d1343-a731-5a90-8730-7bf8d89bf3eb"
version = "0.4.2"

    [deps.SolarPosition.extensions]
    SolarPositionMakieExt = "Makie"
    SolarPositionModelingToolkitExt = ["ModelingToolkit", "Symbolics"]
    SolarPositionOhMyThreadsExt = "OhMyThreads"

    [deps.SolarPosition.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
    OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.SortTileRecursiveTree]]
deps = ["AbstractTrees", "Extents", "GeoInterface"]
git-tree-sha1 = "f9aa6616a9b3bd01f93f27c010f1d25fc5a094a9"
uuid = "746ee33f-1797-42c2-866d-db2fce69d14d"
version = "0.1.4"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5acc6a41b3082920f79ca3c759acbcecf18a8d78"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StateSelection]]
deps = ["BipartiteGraphs", "DocStringExtensions", "FindFirstFunctions", "Graphs", "LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "6fd475ffd04002a88cd14c896a9f09f788ef69d1"
uuid = "64909d44-ed92-46a8-bbd9-f047dfbdc84b"
version = "1.4.0"

    [deps.StateSelection.extensions]
    StateSelectionDeepDiffsExt = "DeepDiffs"

    [deps.StateSelection.weakdeps]
    DeepDiffs = "ab62b9b5-e342-54a8-a765-a90f495de1a6"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "49440414711eddc7227724ae6e570c7d5559a086"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.3.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "SciMLPublic", "Static"]
git-tree-sha1 = "aa1ea41b3d45ac449d10477f65e2b40e3197a0d2"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.9.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0f529006004a8be48f1be25f3451186579392d47"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.17"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "83151ba8065a73f53ca2ae98bc7274d817aa30f2"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.8"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "28145feabf717c5d65c1d5e09747ee7b1ff3ed13"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.3"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "94c58884e013efff548002e8dc2fdd1cb74dfce5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.46"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

[[deps.SymbolicLimits]]
deps = ["SymbolicUtils", "TermInterface"]
git-tree-sha1 = "5085671d2cba1eb02136a3d6661c583e801984c1"
uuid = "19f23fe9-fdab-4a78-91af-e7b7767979c3"
version = "1.1.0"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "ArrayInterface", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "EnumX", "ExproniconLite", "LinearAlgebra", "MacroTools", "Moshi", "MultivariatePolynomials", "MutableArithmetics", "NaNMath", "PrecompileTools", "ReadOnlyArrays", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "TaskLocalValues", "TermInterface", "WeakCacheSets"]
git-tree-sha1 = "df93b5c3b6182a22ee87f8c0d0e404fe3d8ecb9c"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "4.19.0"

    [deps.SymbolicUtils.extensions]
    SymbolicUtilsChainRulesCoreExt = "ChainRulesCore"
    SymbolicUtilsDistributionsExt = "Distributions"
    SymbolicUtilsLabelledArraysExt = "LabelledArrays"
    SymbolicUtilsReverseDiffExt = "ReverseDiff"

    [deps.SymbolicUtils.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.Symbolics]]
deps = ["ADTypes", "AbstractPlutoDingetjes", "ArrayInterface", "Bijections", "CommonWorldInvalidations", "ConstructionBase", "DataStructures", "DiffRules", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "Moshi", "MultivariatePolynomials", "MutableArithmetics", "NaNMath", "PrecompileTools", "Preferences", "Primes", "RecipesBase", "Reexport", "RuntimeGeneratedFunctions", "SciMLPublic", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "SymbolicLimits", "SymbolicUtils", "TermInterface"]
git-tree-sha1 = "02687bc18b509620a6472cc90f65da9d3f885a2f"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "7.15.3"

    [deps.Symbolics.extensions]
    SymbolicsD3TreesExt = "D3Trees"
    SymbolicsDistributionsExt = "Distributions"
    SymbolicsForwardDiffExt = "ForwardDiff"
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsHypergeometricFunctionsExt = "HypergeometricFunctions"
    SymbolicsLatexifyExt = ["Latexify", "LaTeXStrings"]
    SymbolicsLuxExt = "Lux"
    SymbolicsNemoExt = "Nemo"
    SymbolicsPreallocationToolsExt = ["PreallocationTools", "ForwardDiff"]
    SymbolicsSymPyExt = "SymPy"
    SymbolicsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Symbolics.weakdeps]
    D3Trees = "e3df1716-f71e-5df9-9e2d-98e193103c45"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    HypergeometricFunctions = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
    Nemo = "2edaba10-b0f1-5616-af89-8c11ac63239a"
    PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TZJData]]
deps = ["Artifacts"]
git-tree-sha1 = "72df96b3a595b7aab1e101eb07d2a435963a97e2"
uuid = "dc5dba14-91b3-4cab-a142-028a31da12f7"
version = "1.5.0+2025b"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TermInterface]]
git-tree-sha1 = "d673e0aca9e46a2f63720201f55cc7b3e7169b16"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "2.0.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TestItemRunner]]
deps = ["Pkg", "TOML", "Test", "TestItems", "UUIDs"]
git-tree-sha1 = "76f275a3f3d83ece88ec69d73058048de5acb1dc"
uuid = "f8b46487-2199-4994-9208-9a1283c18c0a"
version = "1.1.4"

[[deps.TestItems]]
git-tree-sha1 = "42fd9023fef18b9b78c8343a4e2f3813ffbcefcb"
uuid = "1c621080-faea-4a02-84b6-bbd5e436b8fe"
version = "1.0.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "d969183d3d244b6c33796b5ed01ab97328f2db85"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.5"

[[deps.ThreadsX]]
deps = ["Accessors", "ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "SplittablesBase", "Transducers"]
git-tree-sha1 = "70bd8244f4834d46c3d68bd09e7792d8f571ef04"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.12"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "08c10bc34f4e7743f530793d0985bf3c254e193d"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.8"

[[deps.TimeZones]]
deps = ["Artifacts", "Dates", "Downloads", "InlineStrings", "Mocking", "Printf", "Scratch", "TZJData", "Unicode", "p7zip_jll"]
git-tree-sha1 = "d422301b2a1e294e3e4214061e44f338cafe18a2"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.22.2"
weakdeps = ["RecipesBase"]

    [deps.TimeZones.extensions]
    TimeZonesRecipesBaseExt = "RecipesBase"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3748bd928e68c7c346b52125cf41fff0de6937d0"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.29"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"
weakdeps = ["ConstructionBase", "ForwardDiff", "InverseFunctions", "LaTeXStrings", "Latexify", "NaNMath", "Printf"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

[[deps.WeakCacheSets]]
git-tree-sha1 = "386050ae4353310d8ff9c228f83b1affca2f7f38"
uuid = "d30d5f5c-d141-4870-aa07-aabb0f5fe7d5"
version = "0.1.0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.dlfcn_win32_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e141d67ffe550eadfb5af1bdbdaf138031e4805f"
uuid = "c4b69c83-5512-53e3-94e6-de98773c479f"
version = "1.4.2+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "13b760f97c6e753b47df30cb438d4dc3b50df282"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.5+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e015f211ebb898c8180887012b938f3851e719ac"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.55+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╟─e4f5a6b7-0005-0005-0005-000000000001
# ╟─e4f5a6b7-0005-0005-0005-000000000002
# ╟─e4f5a6b7-0005-0005-0005-000000000003
# ╟─e4f5a6b7-0005-0005-0005-000000000004
# ╟─e4f5a6b7-0005-0005-0005-000000000005
# ╟─e4f5a6b7-0005-0005-0005-000000000006
# ╟─e4f5a6b7-0005-0005-0005-000000000060
# ╟─e4f5a6b7-0005-0005-0005-000000000007
# ╠═e4f5a6b7-0005-0005-0005-000000000008
# ╟─e4f5a6b7-0005-0005-0005-00000000000a
# ╟─e4f5a6b7-0005-0005-0005-00000000000b
# ╠═e4f5a6b7-0005-0005-0005-00000000000c
# ╟─e4f5a6b7-0005-0005-0005-00000000000d
# ╠═e4f5a6b7-0005-0005-0005-00000000000e
# ╟─e4f5a6b7-0005-0005-0005-000000000064
# ╠═e4f5a6b7-0005-0005-0005-00000000000f
# ╠═e4f5a6b7-0005-0005-0005-000000000010
# ╟─e4f5a6b7-0005-0005-0005-000000000011
# ╟─e4f5a6b7-0005-0005-0005-000000000012
# ╟─e4f5a6b7-0005-0005-0005-000000000013
# ╠═e4f5a6b7-0005-0005-0005-000000000014
# ╟─e4f5a6b7-0005-0005-0005-000000000016
# ╟─e4f5a6b7-0005-0005-0005-000000000017
# ╠═e4f5a6b7-0005-0005-0005-000000000040
# ╟─e4f5a6b7-0005-0005-0005-00000000001b
# ╟─e4f5a6b7-0005-0005-0005-00000000001c
# ╟─e4f5a6b7-0005-0005-0005-00000000001d
# ╟─e4f5a6b7-0005-0005-0005-00000000001e
# ╟─e4f5a6b7-0005-0005-0005-00000000001f
# ╠═e4f5a6b7-0005-0005-0005-000000000041
# ╟─e4f5a6b7-0005-0005-0005-000000000026
# ╟─e4f5a6b7-0005-0005-0005-000000000023
# ╠═e4f5a6b7-0005-0005-0005-000000000024
# ╟─e4f5a6b7-0005-0005-0005-000000000065
# ╟─e4f5a6b7-0005-0005-0005-000000000066
# ╠═e4f5a6b7-0005-0005-0005-000000000025
# ╟─e4f5a6b7-0005-0005-0005-000000000027
# ╟─e4f5a6b7-0005-0005-0005-000000000070
# ╠═e4f5a6b7-0005-0005-0005-000000000071
# ╟─e4f5a6b7-0005-0005-0005-000000000073
# ╟─e4f5a6b7-0005-0005-0005-000000000074
# ╟─e4f5a6b7-0005-0005-0005-000000000028
# ╟─e4f5a6b7-0005-0005-0005-000000000029
# ╟─e4f5a6b7-0005-0005-0005-00000000002a
# ╟─e4f5a6b7-0005-0005-0005-00000000002b
# ╟─0b0af5db-880e-4f16-a013-01d9d138c7ec
# ╟─e4f5a6b7-0005-0005-0005-00000000002c
# ╠═e4f5a6b7-0005-0005-0005-00000000002d
# ╠═e4f5a6b7-0005-0005-0005-00000000002e
# ╟─e4f5a6b7-0005-0005-0005-00000000002f
# ╟─e4f5a6b7-0005-0005-0005-000000000030
# ╠═e4f5a6b7-0005-0005-0005-000000000031
# ╟─e4f5a6b7-0005-0005-0005-000000000063
# ╟─e4f5a6b7-0005-0005-0005-000000000032
# ╟─e4f5a6b7-0005-0005-0005-000000000033
# ╠═e4f5a6b7-0005-0005-0005-000000000034
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
