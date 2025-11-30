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
st.caption("Projections utilisateurs, revenus, coûts, trésorerie, scénarios, benchmarks & cap table — v8 corrigée")

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

# ---------------------------------------------------------
# Biomarkers = coûts (pas de revenu)
# ---------------------------------------------------------
st.sidebar.subheader("🧪 Biomarkers (coûts moyens)")

biomarker_cost_avg = st.sidebar.number_input(
    "Coût moyen d'une analyse Biomarkers (€/analyse)",
    0.0, 1000.0, 120.0, 1.0,
    help="Ce que le labo facture à Zolya pour un panel PhenoAge complet (9 biomarkers, logistique, etc.)."
)

biomarker_analyses_per_user_year = st.sidebar.number_input(
    "Nb moyen d'analyses Biomarkers / utilisateur / an",
    0.0, 12.0, 1.0, 0.1,
    help="Moyenne long terme : par ex. 1 panel complet PhenoAge par utilisateur et par an."
)

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
    help="Multiple appliqué au chiffre d'affaires annuel (ARR) pour estimer la pré-money."
)

valuation_year = st.sidebar.slider(
    "Année utilisée pour la valo",
    1, years, min(3, years),
    help="Année de référence pour l'ARR (année n dans la projection)."
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
# FONCTION DE SIMULATION CORRIGÉE
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
    biomarker_cost_avg: float,
    biomarker_analyses_per_user_year: float,
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
    Simulation corrigée avec calculs cohérents du CA et des coûts Biomarkers
    """

    data = []
    users_start = starting_users
    cash = starting_cash

    for m in range(1, months + 1):
        year_index = (m - 1) // 12

        # Inflation salaires
        current_salaries = salaries_monthly * ((1 + salary_inflation_yearly) ** year_index)

        # CAPEX annuel
        current_month_in_year = (m - 1) % 12 + 1
        capex = yearly_capex if current_month_in_year == capex_month else 0.0

        # Logistique - croissance organique
        if max_users > 0:
            logistic_new = logistic_r * users_start * (1 - users_start / max_users)
        else:
            logistic_new = 0.0
        logistic_new = max(logistic_new, 0.0)

        # Acquisition marketing
        if cac > 0:
            new_from_marketing = monthly_marketing_budget / cac
        else:
            new_from_marketing = 0.0

        new_customers = logistic_new + new_from_marketing

        # Churn
        churn = users_start * churn_monthly

        # Update users
        users_end = users_start + new_customers - churn
        users_end = max(users_end, 0.0)
        
        # Application de la limite du marché
        if max_users > 0:
            users_end = min(users_end, max_users)
            saturation_ratio = users_end / max_users
        else:
            saturation_ratio = np.nan

        # Mix Basic / Premium
        premium_users = users_end * premium_share
        basic_users = users_end - premium_users

        # CALCUL DU CA MENSUEL CORRIGÉ
        rev_basic = basic_users * price_basic
        rev_premium = premium_users * price_premium
        revenue_total = rev_basic + rev_premium

        # CALCUL COÛT BIOMARKERS MENSUEL CORRIGÉ
        bio_cost_per_user_month = biomarker_cost_avg * (biomarker_analyses_per_user_year / 12.0)
        cost_biomarkers = users_end * bio_cost_per_user_month

        # Frais paiement
        payment_fees = revenue_total * payment_fee_pct

        # Coûts fixes
        fixed_costs = current_salaries + rent_monthly + tools_monthly + other_fixed_monthly

        # Marketing
        total_marketing = monthly_marketing_budget

        # Total coûts
        total_costs = fixed_costs + cost_biomarkers + payment_fees + total_marketing + capex

        # Cash flow
        cash_flow = revenue_total - total_costs
        cash = cash + cash_flow

        # Unit economics
        if users_end > 0:
            sub_arpu_month = revenue_total / users_end
            psp_fees_per_user_month = payment_fees / users_end
        else:
            sub_arpu_month = 0.0
            psp_fees_per_user_month = 0.0

        gross_margin_per_user_month = sub_arpu_month - bio_cost_per_user_month - psp_fees_per_user_month

        if churn_monthly > 0:
            ltv_approx = gross_margin_per_user_month * (1.0 / churn_monthly)
        else:
            ltv_approx = 0.0

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
                "Saturation_ratio": saturation_ratio,
                "Basic_users": basic_users,
                "Premium_users": premium_users,
                "Rev_basic": rev_basic,
                "Rev_premium": rev_premium,
                "CA_total": revenue_total,
                "Cost_biomarkers": cost_biomarkers,
                "Bio_cost_per_user_month": bio_cost_per_user_month,
                "Payment_fees": payment_fees,
                "PSP_fees_per_user_month": psp_fees_per_user_month,
                "Fixed_costs": fixed_costs,
                "Marketing_costs": total_marketing,
                "Capex": capex,
                "Total_costs": total_costs,
                "Cash_flow": cash_flow,
                "Cash": cash,
                "Sub_ARPU_month": sub_arpu_month,
                "Gross_margin_per_user_month": gross_margin_per_user_month,
                "LTV_approx": ltv_approx,
            }
        )

        users_start = users_end

    df = pd.DataFrame(data)
    return df

# =========================================================
# SCÉNARIOS : SAFE / BASE / MOONSHOT
# =========================================================
def get_scenario_inputs(name: str):
    if name == "Safe":
        return {"churn_delta": +0.02, "cac_mult": 1.3, "mkt_mult": 0.7}
    elif name == "Moon":
        return {"churn_delta": -0.02, "cac_mult": 0.7, "mkt_mult": 1.3}
    else:
        return {"churn_delta": 0.0, "cac_mult": 1.0, "mkt_mult": 1.0}


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
        biomarker_cost_avg=biomarker_cost_avg,
        biomarker_analyses_per_user_year=biomarker_analyses_per_user_year,
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

# CALCUL ANNUEL CORRIGÉ - Prendre la somme des CA mensuels pour l'année
def calculate_yearly_metrics(df):
    yearly_data = []
    for year in range(1, years + 1):
        year_data = df[df['Année'] == year]
        if not year_data.empty:
            yearly_data.append({
                'Année': year,
                'Users_end': year_data['Users_end'].iloc[-1],  # Dernier mois de l'année
                'CA_total': year_data['CA_total'].sum(),  # SOMME des CA mensuels
                'Total_costs': year_data['Total_costs'].sum(),
                'Cash_flow': year_data['Cash_flow'].sum(),
                'Cash_end': year_data['Cash'].iloc[-1],
                'Capex_total': year_data['Capex'].sum(),
                'Bio_costs_total': year_data['Cost_biomarkers'].sum(),
            })
    return pd.DataFrame(yearly_data)

df_base = dfs["Base"]
yearly_base = calculate_yearly_metrics(df_base)

# =========================================================
# TABS
# =========================================================
tab_overview, tab_users, tab_costs, tab_pricing, tab_scenarios, tab_valuation, tab_bench, tab_raw = st.tabs(
    [
        "🏠 Overview",
        "👥 Users & Revenues", 
        "💸 Costs & Cash",
        "🧮 Pricing Sensitivity",
        "🧪 Scenarios",
        "🏦 Valuation & Cap table",
        "📊 Benchmarks",
        "📑 Données brutes & justifs",
    ]
)

# ---------------------------------------------------------
# TAB 1 — OVERVIEW (CORRIGÉ)
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Vue d'ensemble — scénario Base")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        users_year1 = yearly_base.loc[0, 'Users_end']
        st.metric("Utilisateurs fin année 1", f"{int(users_year1):,}".replace(",", " "))
    
    with col2:
        # CA ANNUEL = somme des CA mensuels de l'année
        ca_year1 = yearly_base.loc[0, 'CA_total']
        st.metric("CA année 1 (Base, €)", f"{int(ca_year1):,}".replace(",", " "))
    
    with col3:
        # BURN MOYEN MENSUEL = cash flow total de l'année / 12
        cash_flow_year1 = yearly_base.loc[0, 'Cash_flow']
        burn_mensuel_moyen = cash_flow_year1 / 12
        st.metric("Burn moyen / mois année 1 (Base, €)", f"{int(burn_mensuel_moyen):,}".replace(",", " "))
    
    with col4:
        cash_final = yearly_base.iloc[-1]['Cash_end']
        st.metric("Trésorerie fin horizon (Base, €)", f"{int(cash_final):,}".replace(",", " "))

    st.markdown("----")
    
    # Diagnostic de cohérence
    st.markdown("### 🔍 Diagnostic de cohérence")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.write("**Année 1 - Vérification:**")
        st.write(f"- Utilisateurs fin année 1: {int(users_year1):,}")
        st.write(f"- CA total année 1: {int(ca_year1):,} €")
        st.write(f"- Cash flow année 1: {int(cash_flow_year1):,} €")
        st.write(f"- Burn mensuel moyen: {int(burn_mensuel_moyen):,} €")
        
        # Vérification Biomarkers
        bio_year1 = yearly_base.loc[0, 'Bio_costs_total']
        st.write(f"- Coûts Biomarkers année 1: {int(bio_year1):,} €")
    
    with col_d2:
        st.write("**Dernier mois - Vérification:**")
        last_month = df_base.iloc[-1]
        st.write(f"- Utilisateurs: {last_month['Users_end']:,.0f}")
        st.write(f"- CA mensuel: {last_month['CA_total']:,.0f} €")
        st.write(f"- Coût Biomarkers mensuel: {last_month['Cost_biomarkers']:,.0f} €")
        st.write(f"- Coût Biomarkers/user/mois: {last_month['Bio_cost_per_user_month']:.2f} €")

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        fig_users = px.line(
            df_base, x="Mois", y="Users_end",
            title="Utilisateurs actifs (fin de mois) — Base",
        )
        st.plotly_chart(fig_users, use_container_width=True)

    with col_o2:
        fig_rev = px.line(
            df_base, x="Mois", y="CA_total",
            title="Chiffre d'affaires mensuel (€) — Base",
        )
        st.plotly_chart(fig_rev, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — USERS & REVENUES
# ---------------------------------------------------------
with tab_users:
    st.subheader("👥 Utilisateurs & Revenus — scénario Base")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        fig_users2 = px.line(
            df_base, x="Mois", y=["Users_start", "Users_end"],
            title="Utilisateurs début vs fin de mois — Base",
        )
        st.plotly_chart(fig_users2, use_container_width=True)

    with col_u2:
        fig_sat = px.line(
            df_base, x="Mois", y="Saturation_ratio",
            title="Saturation (%) par rapport au marché max",
        )
        st.plotly_chart(fig_sat, use_container_width=True)

    st.markdown("### Revenus par type (Base)")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_rev_comp = px.line(
            df_base, x="Mois", y=["Rev_basic", "Rev_premium"],
            title="Décomposition des revenus mensuels — Base",
        )
        st.plotly_chart(fig_rev_comp, use_container_width=True)

    with col_r2:
        last_row = df_base.iloc[-1]
        st.metric("Rev. Basic (dernier mois)", f"{int(last_row['Rev_basic']):,} €".replace(",", ' '))
        st.metric("Rev. Premium (dernier mois)", f"{int(last_row['Rev_premium']):,} €".replace(",", ' '))

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
            df_base, x="Mois", y="Cash",
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
                "Bio_costs_total": "{:,.0f}",
            }
        )
    )

    st.markdown("### Unit economics & LTV (après Biomarkers + PSP)")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        fig_unit = px.line(
            df_base,
            x="Mois",
            y=["Sub_ARPU_month", "Bio_cost_per_user_month", "Gross_margin_per_user_month"],
            title="ARPU vs coût Biomarkers vs marge (€/user/mois)",
        )
        st.plotly_chart(fig_unit, use_container_width=True)

    with col_l2:
        last = df_base.iloc[-1]
        arpu_last = last["Sub_ARPU_month"]
        bio_cost_last = last["Bio_cost_per_user_month"]
        margin_last = last["Gross_margin_per_user_month"]

        st.metric("ARPU abonnements (dernier mois)", f"{arpu_last:,.2f} €".replace(",", " "))
        st.metric("Coût Biomarkers / user / mois", f"{bio_cost_last:,.2f} €".replace(",", " "))
        st.metric("Marge nette / user / mois", f"{margin_last:,.2f} €".replace(",", " "))

        if arpu_last > 0:
            bio_vs_arpu = bio_cost_last / arpu_last
            margin_vs_arpu = margin_last / arpu_last
            st.write(f"Poids Biomarkers / ARPU ≈ {bio_vs_arpu*100:,.1f} %")
            st.write(f"Marge nette / ARPU ≈ {margin_vs_arpu*100:,.1f} %")

    last_ltv = df_base["LTV_approx"].iloc[-1]
    st.metric("LTV (approx., marge / churn)", f"{int(last_ltv):,} €".replace(",", " "))
    st.metric("CAC (input, Base)", f"{cac:.0f} €")
    if cac > 0:
        ltv_cac_ratio = last_ltv / cac
        st.write(f"LTV / CAC ≈ {ltv_cac_ratio:.1f}x")

# ---------------------------------------------------------
# TAB 4 — PRICING SENSITIVITY (BREAK-EVEN)
# ---------------------------------------------------------
with tab_pricing:
    st.subheader("🧮 Sensibilité Prix Basic / Premium → rentabilité par utilisateur")

    st.markdown("""
    Cette sensibilité calcule, pour une grille de prix Basic/Premium, 
    la **marge nette moyenne par utilisateur**, en tenant compte :
    - du coût moyen d'un panel Biomarkers (PhenoAge),
    - de la fréquence moyenne d'analyses / an,
    - des frais de paiement,
    - de la répartition Basic / Premium.
    L'objectif : **trouver la zone de prix où Zolya est rentable par utilisateur**.
    """)

    # Grille de sensi
    basic_grid = np.linspace(5, 40, 30)
    premium_grid = np.linspace(10, 80, 30)

    margin_matrix = []
    bio_cost_per_user_month = biomarker_cost_avg * (biomarker_analyses_per_user_year / 12.0)

    for pb in basic_grid:
        row = []
        for pp in premium_grid:
            arpu = pb * (1 - premium_share) + pp * premium_share
            stripe_fee = arpu * payment_fee_pct
            margin = arpu - bio_cost_per_user_month - stripe_fee
            row.append(margin)
        margin_matrix.append(row)

    margin_df = pd.DataFrame(
        margin_matrix,
        index=[f"{pb:.1f}€" for pb in basic_grid],
        columns=[f"{pp:.1f}€" for pp in premium_grid],
    )

    fig_heat = px.imshow(
        margin_df,
        labels=dict(x="Prix Premium (€/mois)", y="Prix Basic (€/mois)", color="Marge nette €/user/mois"),
        aspect="auto",
        color_continuous_scale="RdYlGn",
        origin="lower",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    margin_np = np.array(margin_matrix)
    mask_positive = margin_np >= 0

    if np.any(mask_positive):
        idx = np.where(mask_positive)
        pb_min = basic_grid[idx[0][0]]
        pp_min = premium_grid[idx[1][0]]

        st.success(
            f"✅ **Prix minimum (approx.) pour marge nette ≥ 0**\n\n"
            f"- Basic ≈ **{pb_min:.2f} € / mois**\n"
            f"- Premium ≈ **{pp_min:.2f} € / mois**\n\n"
            f"(donné ton mix Basic/Premium actuel & les coûts Biomarkers saisis)."
        )
    else:
        st.error(
            "❌ Avec les coûts Biomarkers et les frais de paiement actuels, "
            "aucune combinaison Basic/Premium dans la grille ne rend la marge nette positive."
        )

# ---------------------------------------------------------
# TAB 5 — SCENARIOS
# ---------------------------------------------------------
with tab_scenarios:
    st.subheader("🧪 Comparaison de scénarios Safe / Base / Moonshot")

    yearly_all = []
    for scen in scenarios:
        tmp = calculate_yearly_metrics(dfs[scen])
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

# ---------------------------------------------------------
# TAB 6 — VALUATION & CAP TABLE
# ---------------------------------------------------------
with tab_valuation:
    st.subheader("🏦 Valorisation & Cap Table pour la levée (scénario Base)")

    # Trouver l'ARR pour l'année de valorisation
    arr_year_data = yearly_base[yearly_base["Année"] == valuation_year]
    if not arr_year_data.empty:
        arr_valo = arr_year_data["CA_total"].values[0]
    else:
        # Si l'année de valorisation dépasse l'horizon, prendre la dernière année
        arr_valo = yearly_base.iloc[-1]["CA_total"]

    pre_money = arr_valo * valuation_multiple
    post_money = pre_money + round_size

    if post_money > 0:
        investor_pct = round_size / post_money
    else:
        investor_pct = 0.0

    option_pct = option_pool_post
    founders_pct = max(0.0, 1.0 - investor_pct - option_pct)

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
        st.metric("Pré-money (ARR x multiple)", f"{int(pre_money):,} €".replace(",", " "))
    with col_v3:
        st.metric("Post-money", f"{int(post_money):,} €".replace(",", " "))

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
            {"Pourcentage": "{:,.1f} %", "Valeur (€)": "{:,.0f}", "Parts": "{:,.0f}"}
        )
    )

    st.markdown("### Cap table post-money (après levée & option pool)")
    post_cap_table = pd.DataFrame(
        {
            "Actionnaires": ["Fondateurs", "Investisseurs tour", "Option pool"],
            "Pourcentage": [founders_pct * 100, investor_pct * 100, option_pct * 100],
            "Valeur (€)": [founders_pct * post_money, investor_pct * post_money, option_pct * post_money],
            "Parts": [founders_shares_post, investors_shares_post, esop_shares_post],
        }
    )
    st.dataframe(
        post_cap_table.style.format(
            {"Pourcentage": "{:,.1f} %", "Valeur (€)": "{:,.0f}", "Parts": "{:,.0f}"}
        )
    )

# ---------------------------------------------------------
# TAB 7 — BENCHMARKS
# ---------------------------------------------------------
with tab_bench:
    st.subheader("📊 Benchmarks marché & multiples (indicatifs)")

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

# ---------------------------------------------------------
# TAB 8 — RAW DATA & EXPORT
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
    assumptions = {
        "Horizon_annees": years,
        "Taille_marche_max_users": max_users,
        "Logistic_r": logistic_r,
        "Prix_basic": price_basic,
        "Prix_premium": price_premium,
        "Part_premium": premium_share,
        "Cout_moyen_biomarkers": biomarker_cost_avg,
        "Analyses_par_user_par_an": biomarker_analyses_per_user_year,
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
    st.json(assumptions)

    # Vérification des calculs Biomarkers
    st.markdown("### Vérification calculs Biomarkers")
    bio_cost_per_user_month_calc = biomarker_cost_avg * (biomarker_analyses_per_user_year / 12.0)
    st.write(f"Coût Biomarkers par user par mois = {biomarker_cost_avg} € × ({biomarker_analyses_per_user_year} / 12) = {bio_cost_per_user_month_calc:.2f} €")
    
    if len(df_base) > 0:
        last_bio_cost = df_base.iloc[-1]['Bio_cost_per_user_month']
        st.write(f"Valeur calculée dans le modèle : {last_bio_cost:.2f} €")
        st.write(f"✓ Cohérent : {abs(last_bio_cost - bio_cost_per_user_month_calc) < 0.01}")
