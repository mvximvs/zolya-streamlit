import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Zolya — Business Plan Simulator",
    layout="wide"
)

st.title("📊 Zolya — Business Plan & Financial Simulator")
st.caption("Projections utilisateurs, revenus, coûts, trésorerie, scénarios, benchmarks & cap table — v5")

# =========================================================
# SIDEBAR — HYPOTHÈSES GÉNÉRALES
# =========================================================
st.sidebar.header("⚙️ Hypothèses générales")

# Horizon de projection
years = st.sidebar.slider(
    "Horizon de projection (années)",
    1, 10, 5,
    help="Nombre d'années sur lesquelles tu projettes le business."
)
months = years * 12

# ---------------------------------------------------------
# Taille de marché / Saturation (logistique)
# ---------------------------------------------------------
st.sidebar.subheader("🌍 Marché & saturation")

max_users = st.sidebar.number_input(
    "Taille du marché adressable (nb d'utilisateurs max)",
    1_000, 100_000_000, 500_000, 1_000,
    help="Approximation du nombre maximal d'utilisateurs payants que Zolya peut atteindre (TAM/SAM en users)."
)

logistic_r = st.sidebar.slider(
    "Taux de croissance organique logistique r (%/mois)",
    0.0, 50.0, 8.0, 0.5,
    help="r du modèle logistique : croissance organique max par mois, en % des utilisateurs existants (hors marketing)."
) / 100.0

# ---------------------------------------------------------
# Prix & offres
# ---------------------------------------------------------
st.sidebar.subheader("💰 Prix & Produits")

price_basic = st.sidebar.number_input(
    "Prix abonnement Basic (€/mois)",
    0.0, 500.0, 9.99, 0.1,
    help="Prix par mois pour l'offre d'entrée (ex : suivi de base, coaching limité)."
)

price_premium = st.sidebar.number_input(
    "Prix abonnement Premium (€/mois)",
    0.0, 500.0, 19.99, 0.1,
    help="Prix par mois pour l'offre premium (ex : plus de données, coaching avancé)."
)

premium_share = st.sidebar.slider(
    "Part des utilisateurs en Premium (%)",
    0, 100, 30,
    help="Proportion d'utilisateurs qui prennent l'offre Premium, parmi les actifs."
) / 100.0

biomarker_price = st.sidebar.number_input(
    "Prix analyse Biomarkers (€/analyse)",
    0.0, 1000.0, 79.0, 1.0,
    help="Prix facturé au client final pour une analyse de biomarqueurs."
)

biomarker_cost = st.sidebar.number_input(
    "Coût direct Biomarkers (€/analyse)",
    0.0, 1000.0, 25.0, 1.0,
    help="Coût facturé par le labo / partenaire pour chaque analyse."
)

biomarker_buy_rate_year = st.sidebar.slider(
    "% des utilisateurs qui achètent une analyse / an",
    0, 100, 25,
    help="Proportion d'utilisateurs actifs qui achètent AU MOINS une analyse par an."
) / 100.0

# ---------------------------------------------------------
# Dynamiques clients
# ---------------------------------------------------------
st.sidebar.subheader("👥 Utilisateurs & churn")

starting_users = st.sidebar.number_input(
    "Utilisateurs actifs au démarrage",
    0, 1_000_000, 100, 10,
    help="Base initiale d'utilisateurs payants déjà acquis au lancement de la simulation."
)

churn_monthly = st.sidebar.slider(
    "Churn mensuel (%)",
    0.0, 30.0, 5.0, 0.1,
    help="Pourcentage d'utilisateurs qui résilient chaque mois (sur la base utilisateurs début de mois)."
) / 100.0

# ---------------------------------------------------------
# Marketing / Acquisition
# ---------------------------------------------------------
st.sidebar.subheader("📣 Marketing & Acquisition")

monthly_marketing_budget = st.sidebar.number_input(
    "Budget marketing mensuel (€/mois)",
    0.0, 1_000_000.0, 5_000.0, 100.0,
    help="Montant mensuel dépensé en acquisition (ads, influence, etc.)."
)

cac = st.sidebar.number_input(
    "CAC moyen (€/nouveau client)",
    0.1, 10_000.0, 50.0, 1.0,
    help="Coût moyen pour acquérir un nouveau client payant (Budget marketing / nouveaux clients)."
)

# ---------------------------------------------------------
# Structure salariale (OPEX)
# ---------------------------------------------------------
st.sidebar.subheader("🏢 Structure salariale (Opex)")

founders = st.sidebar.number_input(
    "Nb fondateurs salariés",
    0, 10, 2,
    help="Nombre de fondateurs qui se versent un salaire."
)

founder_salary = st.sidebar.number_input(
    "Salaire brut chargé / fondateur (€/mois)",
    0.0, 50_000.0, 4_000.0, 500.0,
    help="Inclure charges patronales approximatives (brut chargé)."
)

employees = st.sidebar.number_input(
    "Nb salariés non-fondateurs",
    0, 200, 3,
    help="Nombre de salariés hors fondateurs (devs, data, sales, ops...)."
)

employee_salary = st.sidebar.number_input(
    "Salaire brut chargé / employé (€/mois)",
    0.0, 50_000.0, 3_000.0, 500.0,
    help="Salaire mensuel moyen chargé par employé non-fondateur."
)

salaries_monthly = founders * founder_salary + employees * employee_salary

rent_monthly = st.sidebar.number_input(
    "Loyers / bureaux / remote (€/mois)",
    0.0, 100_000.0, 1_000.0, 100.0,
    help="Coûts de bureaux, coworking, etc. (ou équivalent remote)."
)

tools_monthly = st.sidebar.number_input(
    "Outils SaaS / infra / IA (€/mois)",
    0.0, 100_000.0, 1_500.0, 100.0,
    help="Serveurs, APIs IA, outils internes, CRM, etc."
)

other_fixed_monthly = st.sidebar.number_input(
    "Autres coûts fixes (€/mois)",
    0.0, 100_000.0, 1_000.0, 100.0,
    help="Assurance, comptable, frais généraux."
)

# ---------------------------------------------------------
# CAPEX
# ---------------------------------------------------------
st.sidebar.subheader("🏗️ CAPEX")

yearly_capex = st.sidebar.number_input(
    "CAPEX annuel (dev produit, R&D, etc.)",
    0.0, 5_000_000.0, 20_000.0, 1_000.0,
    help="Investissements ponctuels (gros dev produit, refonte app, achat matériel). Mets 0 si tu ne veux pas modéliser ça."
)

capex_month = st.sidebar.selectbox(
    "Mois du CAPEX dans l'année",
    list(range(1, 13)),
    index=0,
    format_func=lambda x: f"M{x}",
    help="Mois auquel le CAPEX est décaissé (par ex. M1 = début d'année)."
)

# ---------------------------------------------------------
# Paramètres financiers
# ---------------------------------------------------------
st.sidebar.subheader("💶 Paramètres financiers")

payment_fee_pct = st.sidebar.slider(
    "Frais de paiement (Stripe, etc.) (% CA)",
    0.0, 10.0, 2.5, 0.1,
    help="Frais facturés par le prestataire de paiement (Stripe, PSP...)."
) / 100.0

salary_inflation_yearly = st.sidebar.slider(
    "Inflation salaires/an (%)",
    0.0, 20.0, 3.0, 0.5,
    help="Augmentation moyenne annuelle de la masse salariale."
) / 100.0

# ---------------------------------------------------------
# Trésorerie de départ
# ---------------------------------------------------------
st.sidebar.subheader("💼 Trésorerie")

starting_cash = st.sidebar.number_input(
    "Trésorerie initiale (€)",
    0.0, 10_000_000.0, 50_000.0, 1_000.0,
    help="Cash en banque au début de la simulation (après tours précédents)."
)

# ---------------------------------------------------------
# Valo & Cap Table
# ---------------------------------------------------------
st.sidebar.subheader("📊 Valorisation & Cap Table (levée)")

valuation_multiple = st.sidebar.slider(
    "Multiple de valorisation sur ARR (x)",
    0.5, 25.0, 4.0, 0.5,
    help="Multiple appliqué au chiffre d'affaires annuel (ARR) pour estimer la pré-money. Monte-le si tu veux un scénario plus agressif."
)

valuation_year = st.sidebar.slider(
    "Année utilisée pour la valo",
    1, years, min(3, years),
    help="Année de référence pour l'ARR (année n dans la projection). Tu peux prendre une année future (ex : année 3) pour une valo forward."
)

round_size = st.sidebar.number_input(
    "Montant levé sur ce tour (€)",
    0.0, 100_000_000.0, 1_000_000.0, 50_000.0,
    help="Montant target de la levée (ticket cumulé de ce tour)."
)

option_pool_post = st.sidebar.slider(
    "Option pool cible post-money (%)",
    0.0, 30.0, 10.0, 1.0,
    help="Pourcentage du capital réservé aux BSPCE / ESOP après la levée."
) / 100.0

pre_shares_total = st.sidebar.number_input(
    "Nombre total de parts avant levée",
    1, 10_000_000, 10_000, 100,
    help="Nombre total de parts sociales ou actions existantes avant ce tour."
)

# =========================================================
# FONCTION DE SIMULATION (logistique + marketing + churn)
# =========================================================
def simulate_business_plan(
    months: int,
    starting_users: float,
    max_users: float,
    logistic_r: float,
    churn_monthly: float,
    monthly_marketing_budget: float,
    cac: float,
    price_basic: float,
    price_premium: float,
    premium_share: float,
    biomarker_price: float,
    biomarker_cost: float,
    biomarker_buy_rate_year: float,
    salaries_monthly: float,
    rent_monthly: float,
    tools_monthly: float,
    other_fixed_monthly: float,
    salary_inflation_yearly: float,
    payment_fee_pct: float,
    starting_cash: float,
    yearly_capex: float,
    capex_month: int,
    scenario_name: str = "Base",
):
    """
    Modèle utilisateurs :
    U_{t+1} = U_t + r * U_t * (1 - U_t/K) + (Marketing / CAC) - churn

    - r = logistic_r
    - K = max_users
    - Churn = U_t * churn_monthly
    """

    data = []
    users_start = starting_users
    cash = starting_cash

    for m in range(1, months + 1):
        year_index = (m - 1) // 12  # 0 pour année 1, 1 pour année 2, etc.

        # Inflation salaires chaque année
        current_salaries = salaries_monthly * ((1 + salary_inflation_yearly) ** year_index)

        # CAPEX : une fois par an au mois choisi
        current_month_in_year = (m - 1) % 12 + 1
        capex = yearly_capex if current_month_in_year == capex_month else 0.0

        # --- Croissance organique logistique ---
        if max_users > 0:
            logistic_new = logistic_r * users_start * (1 - users_start / max_users)
        else:
            logistic_new = 0.0
        logistic_new = max(logistic_new, 0.0)

        # --- Acquisition marketing linéaire ---
        if cac > 0:
            new_from_marketing = monthly_marketing_budget / cac
        else:
            new_from_marketing = 0.0

        # Total nouveaux clients du mois
        new_customers = logistic_new + new_from_marketing

        # --- Churn ---
        churn = users_start * churn_monthly

        # --- Mise à jour des utilisateurs ---
        users_end = users_start + new_customers - churn
        users_end = max(users_end, 0.0)
        if max_users > 0:
            users_end = min(users_end, max_users)

        # Mix Basic / Premium
        premium_users = users_end * premium_share
        basic_users = users_end - premium_users

        # Revenus abonnements
        rev_basic = basic_users * price_basic
        rev_premium = premium_users * price_premium

        # Revenus Biomarkers
        biomarker_users_month = users_end * (biomarker_buy_rate_year / 12.0)
        rev_biomarkers = biomarker_users_month * biomarker_price

        # CA total
        revenue_total = rev_basic + rev_premium + rev_biomarkers

        # Coûts variables
        cost_biomarkers = biomarker_users_month * biomarker_cost
        payment_fees = revenue_total * payment_fee_pct

        # Coûts fixes
        fixed_costs = current_salaries + rent_monthly + tools_monthly + other_fixed_monthly

        # Coût marketing (Opex)
        total_marketing = monthly_marketing_budget

        # Total coûts
        total_costs = fixed_costs + cost_biomarkers + payment_fees + total_marketing + capex

        # Cash flow
        cash_flow = revenue_total - total_costs
        cash = cash + cash_flow

        # LTV approximative
        if users_end > 0:
            arpu_month = (rev_basic + rev_premium) / users_end
        else:
            arpu_month = 0.0

        if churn_monthly > 0:
            ltv = arpu_month * (1.0 / churn_monthly)
        else:
            ltv = 0.0

        data.append(
            {
                "Scenario": scenario_name,
                "Mois": m,
                "Année": year_index + 1,
                "Users_start": users_start,
                "New_customers": new_customers,
                "Logistic_new": logistic_new,
                "New_from_marketing": new_from_marketing,
                "Churn": churn,
                "Users_end": users_end,
                "Basic_users": basic_users,
                "Premium_users": premium_users,
                "Rev_basic": rev_basic,
                "Rev_premium": rev_premium,
                "Rev_biomarkers": rev_biomarkers,
                "CA_total": revenue_total,
                "Cost_biomarkers": cost_biomarkers,
                "Payment_fees": payment_fees,
                "Fixed_costs": fixed_costs,
                "Marketing_costs": total_marketing,
                "Capex": capex,
                "Total_costs": total_costs,
                "Cash_flow": cash_flow,
                "Cash": cash,
                "ARPU_month": arpu_month,
                "LTV_approx": ltv,
            }
        )

        users_start = users_end

    df = pd.DataFrame(data)
    return df


# =========================================================
# SCÉNARIOS : SAFE / BASE / MOONSHOT
# =========================================================
def get_scenario_inputs(name: str):
    """
    SAFE : plus conservateur
    MOON : plus agressif
    """
    if name == "Safe":
        return {
            "churn_delta": +0.02,
            "cac_mult": 1.3,
            "mkt_mult": 0.7,
        }
    elif name == "Moon":
        return {
            "churn_delta": -0.02,
            "cac_mult": 0.7,
            "mkt_mult": 1.3,
        }
    else:  # Base
        return {
            "churn_delta": 0.0,
            "cac_mult": 1.0,
            "mkt_mult": 1.0,
        }


scenarios = ["Safe", "Base", "Moon"]
dfs = {}

for scen in scenarios:
    mods = get_scenario_inputs(scen)

    scen_churn = min(max(churn_monthly + mods["churn_delta"], 0.0), 0.30)
    scen_cac = cac * mods["cac_mult"]
    scen_mkt = monthly_marketing_budget * mods["mkt_mult"]

    df_s = simulate_business_plan(
        months=months,
        starting_users=starting_users,
        max_users=max_users,
        logistic_r=logistic_r,
        churn_monthly=scen_churn,
        monthly_marketing_budget=scen_mkt,
        cac=scen_cac,
        price_basic=price_basic,
        price_premium=price_premium,
        premium_share=premium_share,
        biomarker_price=biomarker_price,
        biomarker_cost=biomarker_cost,
        biomarker_buy_rate_year=biomarker_buy_rate_year,
        salaries_monthly=salaries_monthly,
        rent_monthly=rent_monthly,
        tools_monthly=tools_monthly,
        other_fixed_monthly=other_fixed_monthly,
        salary_inflation_yearly=salary_inflation_yearly,
        payment_fee_pct=payment_fee_pct,
        starting_cash=starting_cash,
        yearly_capex=yearly_capex,
        capex_month=capex_month,
        scenario_name=scen,
    )
    dfs[scen] = df_s

df_base = dfs["Base"]
yearly_base = df_base.groupby("Année").agg(
    Users_end=("Users_end", "last"),
    CA_total=("CA_total", "sum"),
    Total_costs=("Total_costs", "sum"),
    Cash_flow=("Cash_flow", "sum"),
    Cash_end=("Cash", "last"),
    Capex_total=("Capex", "sum"),
).reset_index()

# =========================================================
# ONGLETS DE SORTIE
# =========================================================
tab_overview, tab_users, tab_costs, tab_scenarios, tab_valuation, tab_bench, tab_raw = st.tabs(
    [
        "🏠 Overview",
        "👥 Users & Revenues",
        "💸 Costs & Cash",
        "🧪 Scenarios",
        "🏦 Valuation & Cap table",
        "📊 Benchmarks",
        "📑 Données brutes & justifs",
    ]
)

# ---------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Vue d'ensemble — scénario Base")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Utilisateurs fin année 1",
            f"{int(yearly_base.loc[0, 'Users_end']):,}".replace(",", " ")
        )
    with col2:
        st.metric(
            "CA année 1 (Base, €)",
            f"{int(yearly_base.loc[0, 'CA_total']):,}".replace(",", " ")
        )
    with col3:
        st.metric(
            "Burn moyen / mois année 1 (Base, €)",
            f"{int(yearly_base.loc[0, 'Cash_flow'] / 12):,}".replace(",", " ")
        )
    with col4:
        st.metric(
            "Trésorerie fin horizon (Base, €)",
            f"{int(yearly_base.iloc[-1]['Cash_end']):,}".replace(",", " ")
        )

    st.markdown("----")
    st.markdown("### Courbes principales (Base)")

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        fig_users = px.line(
            df_base,
            x="Mois",
            y="Users_end",
            title="Utilisateurs actifs (fin de mois) — Base",
        )
        st.plotly_chart(fig_users, use_container_width=True)

    with col_o2:
        fig_rev = px.line(
            df_base,
            x="Mois",
            y="CA_total",
            title="Chiffre d'affaires mensuel (€) — Base",
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with st.expander("Explication rapide du modèle (Overview)", expanded=False):
        st.write("""
        - Croissance utilisateurs = **logistique** (r·U·(1−U/K)) + acquisition marketing − churn.
        - Tu contrôles :
            - K = `Taille du marché adressable`,
            - r = `Taux de croissance logistique r`,
            - acquisition = `Budget marketing` / `CAC`,
            - churn = `Churn mensuel`.
        - Les revenus combinent abonnements (Basic/Premium) + Biomarkers.
        """)

# ---------------------------------------------------------
# TAB 2 — USERS & REVENUES
# ---------------------------------------------------------
with tab_users:
    st.subheader("👥 Utilisateurs & Revenus — scénario Base")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        fig_users2 = px.line(
            df_base,
            x="Mois",
            y=["Users_start", "Users_end"],
            title="Utilisateurs début vs fin de mois — Base",
        )
        st.plotly_chart(fig_users2, use_container_width=True)

    with col_u2:
        fig_seg = px.line(
            df_base,
            x="Mois",
            y=["Basic_users", "Premium_users"],
            title="Répartition Basic / Premium — Base",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("### Revenus par type (Base)")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_rev_comp = px.line(
            df_base,
            x="Mois",
            y=["Rev_basic", "Rev_premium", "Rev_biomarkers"],
            title="Décomposition des revenus mensuels — Base",
        )
        st.plotly_chart(fig_rev_comp, use_container_width=True)

    with col_r2:
        last_row = df_base.iloc[-1]
        st.metric("Rev. Basic (dernier mois)", f"{int(last_row['Rev_basic']):,} €".replace(",", ' '))
        st.metric("Rev. Premium (dernier mois)", f"{int(last_row['Rev_premium']):,} €".replace(",", ' '))
        st.metric("Rev. Biomarkers (dernier mois)", f"{int(last_row['Rev_biomarkers']):,} €".replace(",", ' '))

    with st.expander("Justification du modèle utilisateurs / revenus"):
        st.write("""
        - **Terme logistique** : r·U·(1−U/K) = croissance organique qui ralentit en approchant K.
        - **Acquisition marketing** : Budget marketing / CAC, ajouté au terme logistique.
        - **Churn** : U_t·churn, retire des utilisateurs.
        - **Revenus** :
            - Basic/Premium = nb d'utilisateurs * prix mensuel,
            - Biomarkers = % d'utilisateurs/an converti en volume mensuel * prix.
        """)

# ---------------------------------------------------------
# TAB 3 — COSTS & CASH
# ---------------------------------------------------------
with tab_costs:
    st.subheader("💸 Coûts, Opex, CAPEX & Trésorerie — Base")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_costs = px.line(
            df_base,
            x="Mois",
            y=["Fixed_costs", "Marketing_costs", "Cost_biomarkers", "Payment_fees", "Capex"],
            title="Décomposition des coûts mensuels — Base",
        )
        st.plotly_chart(fig_costs, use_container_width=True)

    with col_c2:
        fig_cash = px.line(
            df_base,
            x="Mois",
            y="Cash",
            title="Trésorerie projetée (€) — Base",
        )
        st.plotly_chart(fig_cash, use_container_width=True)

    st.markdown("### Synthèse par année — Base")
    st.dataframe(
        yearly_base.style.format(
            {
                "Users_end": "{:,.0f}",
                "CA_total": "{:,.0f}",
                "Total_costs": "{:,.0f}",
                "Cash_flow": "{:,.0f}",
                "Cash_end": "{:,.0f}",
                "Capex_total": "{:,.0f}",
            }
        )
    )

    with st.expander("Justification de la partie CAPEX / Opex"):
        st.write("""
        - **CAPEX** = dépenses ponctuelles (gros projets produit, infra, R&D). On les décaissent au mois choisi, une fois par an.
        - **Opex** = tous les coûts récurrents : salaires, loyers, outils, marketing, frais de paiement, coûts Biomarkers.
        - Si tu ne veux pas modéliser le CAPEX, mets simplement **0** en CAPEX annuel.
        """)

    st.markdown("### Unit economics — Base")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        fig_arpu = px.line(
            df_base,
            x="Mois",
            y="ARPU_month",
            title="ARPU mensuel (abonnements) (€ / user / mois)",
        )
        st.plotly_chart(fig_arpu, use_container_width=True)

    with col_l2:
        fig_ltv = px.line(
            df_base,
            x="Mois",
            y="LTV_approx",
            title="LTV approximative (€ / utilisateur)",
        )
        st.plotly_chart(fig_ltv, use_container_width=True)

    last_ltv = df_base["LTV_approx"].iloc[-1]
    st.metric("LTV approx. (dernier mois, Base)", f"{int(last_ltv):,} €".replace(",", " "))
    st.metric("CAC (input, Base)", f"{cac:.0f} €")
    if cac > 0:
        ltv_cac_ratio = last_ltv / cac
        st.write(f"**LTV / CAC ≈ {ltv_cac_ratio:.1f}x** (cible classique : > 3x).")

# ---------------------------------------------------------
# TAB 4 — SCENARIOS
# ---------------------------------------------------------
with tab_scenarios:
    st.subheader("🧪 Comparaison de scénarios Safe / Base / Moonshot")

    yearly_all = []
    for scen in scenarios:
        tmp = dfs[scen].groupby("Année").agg(
            Users_end=("Users_end", "last"),
            CA_total=("CA_total", "sum"),
            Total_costs=("Total_costs", "sum"),
            Cash_flow=("Cash_flow", "sum"),
            Cash_end=("Cash", "last"),
        ).reset_index()
        tmp["Scenario"] = scen
        yearly_all.append(tmp)

    yearly_all = pd.concat(yearly_all, ignore_index=True)

    st.markdown("### CA annuel par scénario")
    st.dataframe(
        yearly_all.pivot(index="Année", columns="Scenario", values="CA_total")
        .round(0)
        .style.format("{:,.0f}")
    )

    fig_scen_ca = px.line(
        yearly_all,
        x="Année",
        y="CA_total",
        color="Scenario",
        markers=True,
        title="Comparaison CA annuel par scénario",
    )
    st.plotly_chart(fig_scen_ca, use_container_width=True)

    st.markdown("### Trésorerie fin d'année par scénario")
    fig_scen_cash = px.line(
        yearly_all,
        x="Année",
        y="Cash_end",
        color="Scenario",
        markers=True,
        title="Comparaison trésorerie fin d'année par scénario",
    )
    st.plotly_chart(fig_scen_cash, use_container_width=True)

    with st.expander("Logique des scénarios"):
        st.write("""
        - **Safe** : CAC plus élevé, churn plus fort, budget marketing réduit → trajectoire prudente.
        - **Base** : reflète exactement les hypothèses définies dans la sidebar.
        - **Moon** : CAC plus faible, churn amélioré, budget marketing plus agressif → scénario d'hyper-croissance.
        """)

# ---------------------------------------------------------
# TAB 5 — VALUATION & CAP TABLE
# ---------------------------------------------------------
with tab_valuation:
    st.subheader("🏦 Valorisation & Cap Table pour la levée (scénario Base)")

    # ARR sur l'année choisie pour la valo (scénario Base)
    arr_year = yearly_base.loc[yearly_base["Année"] == valuation_year, "CA_total"]
    if not arr_year.empty:
        arr_valo = arr_year.values[0]
    else:
        arr_valo = yearly_base["CA_total"].iloc[-1]

    pre_money = arr_valo * valuation_multiple
    post_money = pre_money + round_size

    if post_money > 0:
        investor_pct = round_size / post_money
    else:
        investor_pct = 0.0

    option_pct = option_pool_post
    founders_pct = max(0.0, 1.0 - investor_pct - option_pct)

    # Prix par part et structure en parts
    if pre_shares_total > 0:
        price_per_share_pre = pre_money / pre_shares_total
    else:
        price_per_share_pre = 0.0

    if price_per_share_pre > 0:
        new_shares = round_size / price_per_share_pre
    else:
        new_shares = 0.0

    total_shares_post = pre_shares_total + new_shares

    founders_shares_post = total_shares_post * founders_pct
    investors_shares_post = total_shares_post * investor_pct
    esop_shares_post = total_shares_post * option_pct

    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.metric(
            f"ARR année {valuation_year} (Base)",
            f"{int(arr_valo):,} €".replace(",", " ")
        )
    with col_v2:
        st.metric(
            "Pré-money (ARR x multiple)",
            f"{int(pre_money):,} €".replace(",", " ")
        )
    with col_v3:
        st.metric(
            "Post-money",
            f"{int(post_money):,} €".replace(",", " ")
        )

    st.markdown(
        "💡 *Pré-money = ARR année choisie × multiple. "
        "Si la valo te paraît basse, monte le multiple ou utilise une année plus tardive (forward ARR).*"
    )

    st.markdown("### Cap table pré-money (simplifiée)")
    pre_cap_table = pd.DataFrame(
        {
            "Actionnaires": ["Fondateurs"],
            "Pourcentage": [100.0],
            "Valeur (€)": [pre_money],
            "Parts": [pre_shares_total],
        }
    )
    st.dataframe(
        pre_cap_table.style.format(
            {
                "Pourcentage": "{:,.1f} %",
                "Valeur (€)": "{:,.0f}",
                "Parts": "{:,.0f}",
            }
        )
    )

    st.markdown("### Cap table post-money (après levée & création option pool)")
    post_cap_table = pd.DataFrame(
        {
            "Actionnaires": ["Fondateurs", "Investisseurs tour", "Option pool"],
            "Pourcentage": [
                founders_pct * 100,
                investor_pct * 100,
                option_pct * 100,
            ],
            "Valeur (€)": [
                founders_pct * post_money,
                investor_pct * post_money,
                option_pct * post_money,
            ],
            "Parts": [
                founders_shares_post,
                investors_shares_post,
                esop_shares_post,
            ],
        }
    )
    st.dataframe(
        post_cap_table.style.format(
            {
                "Pourcentage": "{:,.1f} %",
                "Valeur (€)": "{:,.0f}",
                "Parts": "{:,.0f}",
            }
        )
    )

    fig_cap = px.bar(
        post_cap_table,
        x="Actionnaires",
        y="Pourcentage",
        title="Répartition du capital post-money (simplifiée)",
        text="Pourcentage",
    )
    st.plotly_chart(fig_cap, use_container_width=True)

    with st.expander("Comment pitcher cette partie à un investisseur ?"):
        st.write("""
        - On montre la **trajectoire d'ARR** (CA annuel récurrent) du scénario Base.
        - On applique un multiple (par ex. 4x, 6x, 8x) pour obtenir une **pré-money**.
        - On ajoute le montant levé → **post-money**.
        - On construit la **cap table post-tour** avec :
            - Fondateurs,
            - Nouveaux investisseurs,
            - Option pool BSPCE.
        - Si tu veux une valo plus haute :
            - choisir une année avec plus d'ARR (année 3 ou 4),
            - augmenter le multiple (pour refléter AI / health premium),
            - ou modéliser un plan plus agressif (onglet scénarios).
        """)

# ---------------------------------------------------------
# TAB 6 — BENCHMARKS
# ---------------------------------------------------------
with tab_bench:
    st.subheader("📊 Benchmarks marché & multiples (indicatifs)")

    st.markdown("### Taille de marché (ordre de grandeur, à adapter)")
    market_df = pd.DataFrame(
        {
            "Segment": [
                "Bien-être / santé préventive (France)",
                "Health & wellness global",
                "Wellness apps (global)",
            ],
            "Ordre de grandeur": [
                "≈ 30–40 Md€",
                "≈ 3 500–5 500 Md$ (2023–2030)",
                "≈ 10–15 Md$ (avec fort CAGR)",
            ],
            "Commentaire": [
                "Inclut bien-être, coaching, soins non médicaux.",
                "Inclut nutrition, fitness, mental health, etc.",
                "Cible directe de produits type Zolya (app + IA + data santé).",
            ],
        }
    )
    st.dataframe(market_df)

    st.markdown("### Multiples de valorisation ARR (indicatifs, non contractuels)")
    mult_df = pd.DataFrame(
        {
            "Type d'actif / secteur": [
                "SaaS B2B moyen",
                "SaaS HealthTech / MedTech en croissance",
                "Wellness app grand public",
                "AI health infra / high growth",
            ],
            "Fourchette multiple ARR": [
                "2–5x",
                "3–8x",
                "1–4x",
                "6–15x+",
            ],
            "Commentaire": [
                "Selon croissance, marge, churn.",
                "Plus haute si forte croissance et rétention.",
                "Dépend beaucoup du brand / rétention.",
                "Cas bull pour pitch agressif.",
            ],
        }
    )
    st.dataframe(mult_df)

    st.markdown("""
    👉 Pour Zolya, tu peux justifier :
    - un multiple **conservateur** 3–4x si tu restes prudent,
    - un multiple **plus agressif** 6–8x si tu positionnes Zolya comme AI-powered health analytics / biomarker stack
      avec forte croissance, moat data et intégration partenaires.
    """)

# ---------------------------------------------------------
# TAB 7 — RAW DATA & EXPORT
# ---------------------------------------------------------
with tab_raw:
    st.subheader("📑 Données brutes — scénario Base")

    st.write("Aperçu des 24 premiers mois (Base) :")
    st.dataframe(df_base.head(24))

    st.markdown("---")
    st.subheader("📤 Export des données")

    csv = df_base.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger les projections mensuelles (Base) en CSV",
        data=csv,
        file_name="zolya_bp_projections_mensuelles_base.csv",
        mime="text/csv",
    )

    st.markdown("### Rappel des principales hypothèses saisies")
    st.json(
        {
            "Horizon_annees": years,
            "Taille_marche_max_users": max_users,
            "Logistic_r": logistic_r,
            "Prix_basic": price_basic,
            "Prix_premium": price_premium,
            "Part_premium": premium_share,
            "Prix_biomarkers": biomarker_price,
            "Cout_biomarkers": biomarker_cost,
            "%_users_achetant_biomarkers/an": biomarker_buy_rate_year,
            "Starting_users": starting_users,
            "Churn_mensuel": churn_monthly,
            "Budget_marketing": monthly_marketing_budget,
            "CAC": cac,
            "Masse_salariale_mensuelle": salaries_monthly,
            "CAPEX_annuel": yearly_capex,
            "Frais_paiement_%CA": payment_fee_pct,
            "Inflation_salaires/an": salary_inflation_yearly,
            "Tresorerie_initiale": starting_cash,
            "Valo_multiple_ARR": valuation_multiple,
            "Valo_annee_ref": valuation_year,
            "Montant_leve": round_size,
            "Option_pool_post": option_pool_post,
            "Parts_pre_money": pre_shares_total,
        }
    )
