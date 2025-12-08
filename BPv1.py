import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Zolya — Business Plan Simulator",
    layout="wide"
)

st.title("📊 Zolya — Business Plan & Financial Simulator")
st.caption("Projections utilisateurs, revenus, coûts, trésorerie, scénarios, benchmarks & cap table — v10 avec Burn Curve, Cap Table dynamique, Levée et Structure EU")

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
# TABS MIS À JOUR
# =========================================================
tab_europe, tab_fundraising, tab_overview, tab_users, tab_costs, tab_pricing, tab_scenarios, tab_valuation, tab_captable_dynamic, tab_burn, tab_bench, tab_raw = st.tabs(
    [
        "🇪🇺 Structure Européenne",
        "💰 Levée & Capital", 
        "🏠 Overview",
        "👥 Users & Revenues", 
        "💸 Costs & Cash",
        "🧮 Pricing Sensitivity",
        "🧪 Scenarios",
        "🏦 Valuation & Cap table",
        "📈 Cap Table Dynamique",
        "🔥 Burn & Depletion",
        "📊 Benchmarks",
        "📑 Données brutes & justifs",
    ]
)

# ---------------------------------------------------------
# TAB 1 — STRUCTURE EUROPÉENNE
# ---------------------------------------------------------
with tab_europe:
    st.subheader("🇪🇺 Gestion de Trésorerie & Structure Holding Européenne")
    
    st.markdown("""
    ### 📋 Vision Future : Structure Multi-Entités
    
    **Architecture proposée :**
    1. **Holding France** : Propriétaire de l'IP, stratégie groupe
    2. **OpCo France** : Opérations commerciales France
    3. **OpCo Allemagne** : Expansion DACH region
    4. **OpCo UK** : Marché anglophone
    5. **OpCo Espagne** : Marché sud-européen
    """)
    
    # Configuration de la structure
    st.markdown("### ⚙️ Configuration des Filiales")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    
    with col_e1:
        fr_revenue_share = st.slider("Part CA France (%)", 0, 100, 60, 5, key="fr_share")
        de_revenue_share = st.slider("Part CA Allemagne (%)", 0, 100, 20, 5, key="de_share")
    
    with col_e2:
        uk_revenue_share = st.slider("Part CA UK (%)", 0, 100, 10, 5, key="uk_share")
        es_revenue_share = st.slider("Part CA Espagne (%)", 0, 100, 10, 5, key="es_share")
    
    with col_e3:
        # Vérification cohérence
        total_share = fr_revenue_share + de_revenue_share + uk_revenue_share + es_revenue_share
        if total_share != 100:
            st.warning(f"Total: {total_share}%. Normaliser à 100%")
        else:
            st.success("Répartition OK")
    
    # Simulation de trésorerie par entité
    st.markdown("### 💰 Simulation de Trésorerie par Entité")
    
    # Créer un dataframe pour la simulation
    months_sim = min(24, months)
    entities = ['Holding', 'France', 'Allemagne', 'UK', 'Espagne']
    
    cash_simulation = []
    for m in range(1, months_sim + 1):
        # Répartition hypothétique des revenus
        total_rev = df_base[df_base['Mois'] == m]['CA_total'].values[0] if m <= len(df_base) else 0
        
        cash_simulation.append({
            'Mois': m,
            'Holding': starting_cash * 0.2,  # 20% dans holding
            'France': (total_rev * fr_revenue_share/100) * 0.8,  # 80% des revenus France
            'Allemagne': (total_rev * de_revenue_share/100) * 0.7,
            'UK': (total_rev * uk_revenue_share/100) * 0.7,
            'Espagne': (total_rev * es_revenue_share/100) * 0.7
        })
    
    cash_df = pd.DataFrame(cash_simulation)
    
    # Graphique de trésorerie par entité
    fig_europe_cash = go.Figure()
    
    colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa', '#ffa15a']
    
    for i, entity in enumerate(entities[1:]):  # Exclure Holding pour plus de clarté
        fig_europe_cash.add_trace(go.Scatter(
            x=cash_df['Mois'],
            y=cash_df[entity],
            name=entity,
            line=dict(color=colors[i % len(colors)], width=2),
            stackgroup='one'  # Pour un graphique empilé
        ))
    
    fig_europe_cash.update_layout(
        title='Trésorerie projetée par filiale (24 mois)',
        xaxis_title='Mois',
        yaxis_title='Trésorerie (€)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_europe_cash, use_container_width=True)
    
    # Tableau de bord de gestion
    st.markdown("### 📊 Tableau de Bord Gestion Holding")
    
    col_hold1, col_hold2, col_hold3, col_hold4 = st.columns(4)
    
    with col_hold1:
        total_cash = cash_df.iloc[-1][entities].sum()
        st.metric("Trésorerie groupe totale", f"{total_cash:,.0f}€")
    
    with col_hold2:
        holding_cash = cash_df.iloc[-1]['Holding']
        st.metric("Cash Holding", f"{holding_cash:,.0f}€")
    
    with col_hold3:
        # Calculer le besoin en cash working capital
        avg_monthly_burn = df_base['Cash_flow'].mean() * -1 if df_base['Cash_flow'].mean() < 0 else 0
        wc_needs = avg_monthly_burn * 3  # 3 mois de runway par entité
        st.metric("Besoin WC (3 mois)", f"{wc_needs:,.0f}€")
    
    with col_hold4:
        # Efficacité cash par marché
        cash_per_market = cash_df.iloc[-1][['France', 'Allemagne', 'UK', 'Espagne']].sum() / 4
        st.metric("Cash moyen/filiale", f"{cash_per_market:,.0f}€")
    
    # Optimisation fiscale et juridique
    st.markdown("### ⚖️ Optimisation Structurelle")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        st.markdown("**Avantages Holding :**")
        st.write("""
        - Consolidation fiscale
        - Optimisation TVA intra-communautaire
        - Mutualisation des services (legal, finance, HR)
        - Gestion centralisée de la trésorerie
        - Effet de levier pour financement
        """)
    
    with col_opt2:
        st.markdown("**Recommandations :**")
        st.write("""
        - Holding en France (régime mère-fille)
        - Facturation intra-groupe au coût
        - Centralisation R&D dans Holding (CIR)
        - Filiales avec capital minimum local
        - Convention de trésorerie groupée
        """)
    
    # Cash pooling simulation
    st.markdown("### 🔄 Simulation Cash Pooling")
    
    cash_pooling_data = []
    for m in range(1, min(13, months_sim + 1)):  # 12 mois max
        month_data = {
            'Mois': m,
            'Excédent France': max(0, cash_df.iloc[m-1]['France'] - 50000),
            'Déficit Allemagne': max(0, 50000 - cash_df.iloc[m-1]['Allemagne']),
            'Transfert Optimal': min(
                max(0, cash_df.iloc[m-1]['France'] - 50000),
                max(0, 50000 - cash_df.iloc[m-1]['Allemagne'])
            )
        }
        cash_pooling_data.append(month_data)
    
    pooling_df = pd.DataFrame(cash_pooling_data)
    
    fig_pooling = go.Figure()
    
    fig_pooling.add_trace(go.Bar(
        x=pooling_df['Mois'],
        y=pooling_df['Excédent France'],
        name='Excédent France',
        marker_color='green'
    ))
    
    fig_pooling.add_trace(go.Bar(
        x=pooling_df['Mois'],
        y=pooling_df['Déficit Allemagne'],
        name='Déficit Allemagne',
        marker_color='red'
    ))
    
    fig_pooling.add_trace(go.Scatter(
        x=pooling_df['Mois'],
        y=pooling_df['Transfert Optimal'],
        name='Transfert Optimal',
        line=dict(color='blue', width=3),
        mode='lines+markers'
    ))
    
    fig_pooling.update_layout(
        title='Optimisation Cash Pooling France-Allemagne (k€)',
        barmode='group',
        xaxis_title='Mois',
        yaxis_title='Montant (k€)'
    )
    
    st.plotly_chart(fig_pooling, use_container_width=True)
    
    # Export pour plan financier
    st.markdown("### 📤 Export pour Plan Financier")
    
    csv_europe = cash_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger simulation trésorerie EU",
        data=csv_europe,
        file_name="zolya_simulation_tresorerie_europe.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# TAB 2 — LEVÉE & CAPITAL
# ---------------------------------------------------------
with tab_fundraising:
    st.subheader("💰 Allocation du Capital Levé - Healthcare B2B2C")
    
    # Données d'allocation typique pour HealthTech B2B2C
    allocation_data = pd.DataFrame({
        'Catégorie': [
            'R&D Produit (40%)',
            'Marketing & Sales (25%)',
            'Équipe & Opérations (20%)',
            'Biomarkers & Labo (10%)',
            'Fonds de roulement (5%)'
        ],
        'Pourcentage': [40, 25, 20, 10, 5],
        'Description': [
            'Développement plateforme, IA, features',
            'Acquisition clients B2B et B2C, branding',
            'Salaires, recrutement, frais généraux',
            'Tests biomarkers, partenariats labo',
            'Trésorerie opérationnelle, imprévus'
        ],
        'Montant (€)': [round_size * 0.40, round_size * 0.25, 
                       round_size * 0.20, round_size * 0.10, round_size * 0.05]
    })
    
    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        # Donut Chart
        fig_donut = px.pie(
            allocation_data,
            values='Pourcentage',
            names='Catégorie',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu,
            title=f"Allocation des {round_size:,.0f}€ levés"
        )
        
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                         'Montant: %{value:.1f}%<br>' +
                         '€%{customdata:,.0f}<extra></extra>',
            customdata=allocation_data['Montant (€)']
        )
        
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col_d2:
        st.markdown("### 🎯 Détail de l'allocation")
        
        for idx, row in allocation_data.iterrows():
            with st.expander(f"{row['Catégorie']} - {row['Montant (€)']:,.0f}€"):
                st.write(f"**Description :** {row['Description']}")
                
                # Détails spécifiques par catégorie
                if "R&D" in row['Catégorie']:
                    st.write("""
                    **Détail :**
                    - 60% : Développeurs full-stack & data scientists
                    - 20% : Infrastructure cloud & sécurité
                    - 15% : R&D biomarkers & algorithmes IA
                    - 5% : Propriété intellectuelle & certifications
                    """)
                elif "Marketing" in row['Catégorie']:
                    st.write("""
                    **Détail :**
                    - 40% : Acquisition B2B (cliniques, entreprises)
                    - 35% : Acquisition B2C (marketing digital)
                    - 15% : Branding & contenu santé
                    - 10% : Partenariats & relations publiques
                    """)
                elif "Équipe" in row['Catégorie']:
                    st.write("""
                    **Détail :**
                    - 50% : Salaires & charges
                    - 30% : Recrutement & formation
                    - 15% : Bureaux & équipements
                    - 5% : Avantages & bien-être
                    """)
                elif "Biomarkers" in row['Catégorie']:
                    st.write("""
                    **Détail :**
                    - 70% : Tests biomarkers & analyses labo
                    - 20% : Recherche & validation scientifique
                    - 10% : Partenariats avec laboratoires
                    """)
                elif "Fonds" in row['Catégorie']:
                    st.write("""
                    **Détail :**
                    - 60% : Trésorerie opérationnelle (3-6 mois)
                    - 30% : Imprévus & opportunités
                    - 10% : Frais bancaires & assurance
                    """)
    
    # Timeline de déploiement
    st.markdown("### 📅 Timeline de déploiement du capital")
    
    timeline_data = {
        'Phase': ['M1-M3', 'M4-M6', 'M7-M12', 'M13-M18', 'M19-M24'],
        'Focus': [
            'Recrutement & R&D initiale',
            'Développement MVP & tests marché',
            'Lancement commercial & acquisition',
            'Scale-up & optimisation',
            'Expansion & internationalisation'
        ],
        'Budget (%)': [25, 20, 30, 15, 10],
        'Principales Dépenses': [
            'Salaires, outils, labo',
            'Dev produit, tests biomarkers',
            'Marketing, CAC, partenariats',
            'Scale infrastructure, recrutement',
            'Nouveaux marchés, R&D avancée'
        ]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df['Budget (€)'] = timeline_df['Budget (%)'] / 100 * round_size
    
    fig_timeline = px.bar(
        timeline_df,
        x='Phase',
        y='Budget (%)',
        hover_data=['Budget (€)', 'Focus', 'Principales Dépenses'],
        color='Budget (%)',
        color_continuous_scale='Viridis',
        title='Déploiement du capital sur 24 mois'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Comparaison avec les benchmarks du secteur
    st.markdown("### 📊 Benchmarks d'allocation HealthTech B2B2C")
    
    benchmark_data = pd.DataFrame({
        'Catégorie': ['R&D', 'Sales & Marketing', 'G&A', 'Clinical/Lab'],
        'Zolya (proposé)': [40, 25, 25, 10],
        'Moyenne secteur': [35, 30, 25, 10],
        'Best-in-class': [45, 25, 20, 10]
    })
    
    fig_bench = go.Figure()
    
    fig_bench.add_trace(go.Bar(
        name='Zolya',
        x=benchmark_data['Catégorie'],
        y=benchmark_data['Zolya (proposé)'],
        marker_color='#636efa'
    ))
    
    fig_bench.add_trace(go.Bar(
        name='Moyenne secteur',
        x=benchmark_data['Catégorie'],
        y=benchmark_data['Moyenne secteur'],
        marker_color='#ef553b'
    ))
    
    fig_bench.add_trace(go.Bar(
        name='Best-in-class',
        x=benchmark_data['Catégorie'],
        y=benchmark_data['Best-in-class'],
        marker_color='#00cc96'
    ))
    
    fig_bench.update_layout(
        title='Comparaison avec les benchmarks du secteur',
        barmode='group',
        yaxis_title='Pourcentage (%)'
    )
    
    st.plotly_chart(fig_bench, use_container_width=True)

# ---------------------------------------------------------
# TAB 3 — OVERVIEW (CORRIGÉ)
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
# TAB 4 — USERS & REVENUES
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
# TAB 5 — COSTS & CASH
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
# TAB 6 — PRICING SENSITIVITY (BREAK-EVEN)
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
# TAB 7 — SCENARIOS
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
# TAB 8 — VALUATION & CAP TABLE
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
# TAB 9 — CAP TABLE DYNAMIQUE (MULTI-ROUNDS)
# ---------------------------------------------------------
with tab_captable_dynamic:
    st.subheader("📈 Cap Table Dynamique avec Dilutions Multi-Rounds")
    
    st.markdown("""
    **Comment ça marche:**
    1. Configure les levées de fonds futures
    2. Les pourcentages s'ajustent automatiquement à chaque dilution
    3. L'option pool peut être reconstitué à chaque levée
    4. Visualisation de l'évolution des parts dans le temps
    """)
    
    # Configuration des tours de levée
    st.markdown("### 🏦 Configuration des tours de levée")
    
    col_round1, col_round2, col_round3 = st.columns(3)
    
    with col_round1:
        st.markdown("**Seed Round**")
        seed_round = st.number_input("Montant Seed (€)", 0.0, 10_000_000.0, 1_000_000.0, 50_000.0, key="seed_round")
        seed_val_mult = st.slider("Multiple valo Seed (x ARR)", 1.0, 10.0, 4.0, 0.5, key="seed_mult")
        seed_year = st.slider("Année Seed", 1, years, 1, key="seed_year")
    
    with col_round2:
        st.markdown("**Series A**")
        series_a = st.number_input("Montant Series A (€)", 0.0, 20_000_000.0, 3_000_000.0, 100_000.0, key="series_a")
        series_a_mult = st.slider("Multiple valo Series A (x ARR)", 2.0, 15.0, 6.0, 0.5, key="series_a_mult")
        series_a_year = st.slider("Année Series A", 2, years, 3, key="series_a_year")
    
    with col_round3:
        st.markdown("**Series B**")
        series_b = st.number_input("Montant Series B (€)", 0.0, 50_000_000.0, 10_000_000.0, 500_000.0, key="series_b")
        series_b_mult = st.slider("Multiple valo Series B (x ARR)", 3.0, 20.0, 8.0, 0.5, key="series_b_mult")
        series_b_year = st.slider("Année Series B", 3, years, 5, key="series_b_year")
    
    # Paramètres généraux
    st.markdown("### ⚙️ Paramètres généraux")
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        initial_esop = st.slider("Option pool initial (%)", 0.0, 30.0, 10.0, 0.5) / 100.0
        esop_replenish = st.slider("Reconstitution option pool après levée (%)", 0.0, 15.0, 5.0, 0.5) / 100.0
    
    with col_opt2:
        founders_initial_shares = st.number_input("Parts initiales fondateurs", 1, 10_000_000, 10_000, 100)
        angels_percentage = st.slider("Business Angels initiaux (%)", 0.0, 30.0, 5.0, 0.5) / 100.0
    
    # Fonction pour calculer les dilutions
    def calculate_cap_table_dynamic():
        # Étape 1: Initial (avant Seed)
        total_shares = founders_initial_shares
        founders_shares = total_shares * (1 - initial_esop - angels_percentage)
        angels_shares = total_shares * angels_percentage
        esop_shares = total_shares * initial_esop
        
        cap_history = [{
            'Round': 'Initial',
            'Année': 0,
            'Total Shares': total_shares,
            'Fondateurs': founders_shares / total_shares * 100,
            'Business Angels': angels_shares / total_shares * 100,
            'Option Pool': esop_shares / total_shares * 100,
            'Seed Investors': 0.0,
            'Series A Investors': 0.0,
            'Series B Investors': 0.0,
            'Valorisation (€)': 0,
            'Montant Levé (€)': 0,
            'Price per Share (€)': 0
        }]
        
        current_total_shares = total_shares
        
        # Seed Round
        if seed_round > 0 and seed_year <= years:
            # Trouver l'ARR pour l'année Seed
            arr_seed_data = yearly_base[yearly_base["Année"] == seed_year]
            arr_seed = arr_seed_data["CA_total"].values[0] if not arr_seed_data.empty else 0
            
            pre_money_seed = arr_seed * seed_val_mult
            post_money_seed = pre_money_seed + seed_round
            investor_pct_seed = seed_round / post_money_seed if post_money_seed > 0 else 0
            
            # Ajuster pour option pool
            investor_pct_seed_adj = investor_pct_seed * (1 - esop_replenish)
            esop_new_pct = esop_replenish
            
            # Calculer les nouvelles parts
            price_per_share = pre_money_seed / current_total_shares if current_total_shares > 0 else 0
            new_shares_seed = seed_round / price_per_share if price_per_share > 0 else 0
            
            # Dilution
            dilution_factor = current_total_shares / (current_total_shares + new_shares_seed)
            
            # Mettre à jour les parts
            current_total_shares = current_total_shares + new_shares_seed
            
            founders_shares *= dilution_factor * (1 - esop_new_pct)
            angels_shares *= dilution_factor * (1 - esop_new_pct)
            esop_shares = esop_shares * dilution_factor * (1 - esop_new_pct) + current_total_shares * esop_new_pct
            seed_investors_shares = new_shares_seed * (1 - esop_new_pct)
            
            cap_history.append({
                'Round': 'Seed',
                'Année': seed_year,
                'Total Shares': current_total_shares,
                'Fondateurs': founders_shares / current_total_shares * 100,
                'Business Angels': angels_shares / current_total_shares * 100,
                'Option Pool': esop_shares / current_total_shares * 100,
                'Seed Investors': seed_investors_shares / current_total_shares * 100,
                'Series A Investors': 0.0,
                'Series B Investors': 0.0,
                'Valorisation (€)': post_money_seed,
                'Montant Levé (€)': seed_round,
                'Price per Share (€)': price_per_share
            })
        
        # Series A
        if series_a > 0 and series_a_year <= years:
            # Mettre à jour les parts pour Series A
            arr_series_a_data = yearly_base[yearly_base["Année"] == series_a_year]
            arr_series_a = arr_series_a_data["CA_total"].values[0] if not arr_series_a_data.empty else 0
            
            pre_money_series_a = arr_series_a * series_a_mult
            post_money_series_a = pre_money_series_a + series_a
            investor_pct_series_a = series_a / post_money_series_a if post_money_series_a > 0 else 0
            
            # Ajuster pour option pool
            investor_pct_series_a_adj = investor_pct_series_a * (1 - esop_replenish)
            esop_new_pct_a = esop_replenish
            
            # Calculer les nouvelles parts
            price_per_share_a = pre_money_series_a / current_total_shares if current_total_shares > 0 else 0
            new_shares_a = series_a / price_per_share_a if price_per_share_a > 0 else 0
            
            # Dilution
            dilution_factor_a = current_total_shares / (current_total_shares + new_shares_a)
            
            # Mettre à jour les parts
            current_total_shares = current_total_shares + new_shares_a
            
            founders_shares *= dilution_factor_a * (1 - esop_new_pct_a)
            angels_shares *= dilution_factor_a * (1 - esop_new_pct_a)
            seed_investors_shares *= dilution_factor_a * (1 - esop_new_pct_a)
            esop_shares = esop_shares * dilution_factor_a * (1 - esop_new_pct_a) + current_total_shares * esop_new_pct_a
            series_a_shares = new_shares_a * (1 - esop_new_pct_a)
            
            cap_history.append({
                'Round': 'Series A',
                'Année': series_a_year,
                'Total Shares': current_total_shares,
                'Fondateurs': founders_shares / current_total_shares * 100,
                'Business Angels': angels_shares / current_total_shares * 100,
                'Option Pool': esop_shares / current_total_shares * 100,
                'Seed Investors': seed_investors_shares / current_total_shares * 100,
                'Series A Investors': series_a_shares / current_total_shares * 100,
                'Series B Investors': 0.0,
                'Valorisation (€)': post_money_series_a,
                'Montant Levé (€)': series_a,
                'Price per Share (€)': price_per_share_a
            })
        
        # Series B
        if series_b > 0 and series_b_year <= years:
            # Mettre à jour les parts pour Series B
            arr_series_b_data = yearly_base[yearly_base["Année"] == series_b_year]
            arr_series_b = arr_series_b_data["CA_total"].values[0] if not arr_series_b_data.empty else 0
            
            pre_money_series_b = arr_series_b * series_b_mult
            post_money_series_b = pre_money_series_b + series_b
            investor_pct_series_b = series_b / post_money_series_b if post_money_series_b > 0 else 0
            
            # Ajuster pour option pool
            investor_pct_series_b_adj = investor_pct_series_b * (1 - esop_replenish)
            esop_new_pct_b = esop_replenish
            
            # Calculer les nouvelles parts
            price_per_share_b = pre_money_series_b / current_total_shares if current_total_shares > 0 else 0
            new_shares_b = series_b / price_per_share_b if price_per_share_b > 0 else 0
            
            # Dilution
            dilution_factor_b = current_total_shares / (current_total_shares + new_shares_b)
            
            # Mettre à jour les parts
            current_total_shares = current_total_shares + new_shares_b
            
            founders_shares *= dilution_factor_b * (1 - esop_new_pct_b)
            angels_shares *= dilution_factor_b * (1 - esop_new_pct_b)
            seed_investors_shares *= dilution_factor_b * (1 - esop_new_pct_b)
            series_a_shares *= dilution_factor_b * (1 - esop_new_pct_b)
            esop_shares = esop_shares * dilution_factor_b * (1 - esop_new_pct_b) + current_total_shares * esop_new_pct_b
            series_b_shares = new_shares_b * (1 - esop_new_pct_b)
            
            cap_history.append({
                'Round': 'Series B',
                'Année': series_b_year,
                'Total Shares': current_total_shares,
                'Fondateurs': founders_shares / current_total_shares * 100,
                'Business Angels': angels_shares / current_total_shares * 100,
                'Option Pool': esop_shares / current_total_shares * 100,
                'Seed Investors': seed_investors_shares / current_total_shares * 100,
                'Series A Investors': series_a_shares / current_total_shares * 100,
                'Series B Investors': series_b_shares / current_total_shares * 100,
                'Valorisation (€)': post_money_series_b,
                'Montant Levé (€)': series_b,
                'Price per Share (€)': price_per_share_b
            })
        
        return pd.DataFrame(cap_history)
    
    # Calculer et afficher la cap table dynamique
    cap_table_dynamic = calculate_cap_table_dynamic()
    
    st.markdown("### 📊 Évolution de la Cap Table")
    
    # Tableau principal
    display_cols = ['Round', 'Année', 'Fondateurs', 'Business Angels', 'Seed Investors', 
                   'Series A Investors', 'Series B Investors', 'Option Pool', 
                   'Valorisation (€)', 'Montant Levé (€)', 'Price per Share (€)']
    
    st.dataframe(
        cap_table_dynamic[display_cols].style.format({
            'Fondateurs': '{:.1f}%',
            'Business Angels': '{:.1f}%',
            'Seed Investors': '{:.1f}%',
            'Series A Investors': '{:.1f}%',
            'Series B Investors': '{:.1f}%',
            'Option Pool': '{:.1f}%',
            'Valorisation (€)': '{:,.0f}',
            'Montant Levé (€)': '{:,.0f}',
            'Price per Share (€)': '{:.2f}'
        })
    )
    
    # Graphique d'évolution
    st.markdown("### 📈 Visualisation des dilutions")
    
    if not cap_table_dynamic.empty:
        # Préparer les données pour le graphique
        melt_df = pd.melt(cap_table_dynamic, 
                         id_vars=['Round', 'Année'],
                         value_vars=['Fondateurs', 'Business Angels', 'Seed Investors', 
                                    'Series A Investors', 'Series B Investors', 'Option Pool'],
                         var_name='Categorie', value_name='Pourcentage')
        
        fig_cap_evolution = px.area(melt_df, x='Année', y='Pourcentage', color='Categorie',
                                   title='Évolution des pourcentages de capital',
                                   category_orders={'Categorie': ['Fondateurs', 'Business Angels', 
                                                                  'Seed Investors', 'Series A Investors',
                                                                  'Series B Investors', 'Option Pool']})
        st.plotly_chart(fig_cap_evolution, use_container_width=True)
    
    # Résumé pour les fondateurs
    st.markdown("### 🎯 Impact sur les fondateurs")
    
    if not cap_table_dynamic.empty:
        last_row = cap_table_dynamic.iloc[-1]
        founders_final_pct = last_row['Fondateurs']
        total_val = last_row['Valorisation (€)']
        founders_value = total_val * founders_final_pct / 100
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Part finale fondateurs", f"{founders_final_pct:.1f}%")
        with col_f2:
            st.metric("Valorisation finale", f"{total_val:,.0f} €".replace(",", " "))
        with col_f3:
            st.metric("Valeur des parts fondateurs", f"{founders_value:,.0f} €".replace(",", " "))

# ---------------------------------------------------------
# TAB 10 — BURN & DEPLETION CURVE (AMÉLIORÉ)
# ---------------------------------------------------------
with tab_burn:
    st.subheader("🔥 Courbe de Burn Rate & Depletion - KPI Clarifiés")
    
    st.markdown("""
    ### 📊 Clarification des KPI de Trésorerie
    
    **Différence entre les concepts :**
    
    | Concept | Définition | Formule (simplifiée) | Utilité |
    |---------|------------|----------------------|---------|
    | **Burn Rate** | Dépenses mensuelles nettes (perte) | - (Revenus - Coûts) | Suivi mensuel de la consommation de cash |
    | **Cash Zero Date** | Date où trésorerie atteint 0 | M0 + (Cash / Burn Rate moyen) | Planification des levées de fonds |
    | **Runway** | Nombre de mois avant cash=0 | Cash / Burn Rate moyen | Durée de survie sans levée |
    | **Break-even** | Moment où revenus = coûts | Cumul(Revenus) = Cumul(Coûts) | Point de rentabilité opérationnelle |
    | **Trésorerie à 0 après X mois** | Cash final projeté après X mois | Cash initial + Σ(Cash Flow) | Vision à horizon fixé |
    
    ---
    """)
    
    # Calculer les KPI
    df_base['Burn_Rate'] = -df_base['Cash_flow']  # Burn = cash flow négatif
    df_base['Cumul_CA'] = df_base['CA_total'].cumsum()
    df_base['Cumul_Couts'] = df_base['Total_costs'].cumsum()
    
    # Création d'un dashboard KPI clair
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    
    with col_kpi1:
        avg_burn = df_base['Burn_Rate'].mean()
        st.metric("🔥 Burn Rate moyen", f"{avg_burn:,.0f}€", 
                 help="Dépenses mensuelles nettes moyennes (négatif = perte)")
    
    with col_kpi2:
        current_cash = df_base['Cash'].iloc[-1]
        current_burn = df_base['Burn_Rate'].iloc[-1]
        if current_burn > 0:
            runway = current_cash / current_burn
        else:
            runway = float('inf')
        st.metric("⏳ Runway actuel", f"{runway:.1f} mois" if runway != float('inf') else "∞",
                 help="Mois restants avant cash=0 au rythme actuel")
    
    with col_kpi3:
        # Trouver le mois de break-even (cumulé)
        break_even_idx = df_base[df_base['Cumul_CA'] >= df_base['Cumul_Couts']].index.min()
        if pd.isna(break_even_idx):
            st.metric("⚖️ Break-even", "Jamais", delta="Non atteint")
        else:
            break_even_month = int(break_even_idx) + 1
            st.metric("⚖️ Break-even", f"M{break_even_month}", 
                     delta=f"Année {(break_even_month-1)//12 + 1}")
    
    with col_kpi4:
        # Trésorerie à différents horizons
        horizon_6m = df_base[df_base['Mois'] <= 6]['Cash'].iloc[-1] if len(df_base[df_base['Mois'] <= 6]) > 0 else 0
        horizon_12m = df_base[df_base['Mois'] <= 12]['Cash'].iloc[-1] if len(df_base[df_base['Mois'] <= 12]) > 0 else 0
        st.metric("💰 Trésorerie 12m", f"{horizon_12m:,.0f}€", 
                 delta=f"{horizon_12m - horizon_6m:,.0f}€ vs 6m")
    
    with col_kpi5:
        # Cash zero date
        cash_zero_idx = df_base[df_base['Cash'] <= 0].index.min()
        if pd.isna(cash_zero_idx):
            st.metric("📅 Cash Zero", "Jamais", delta="Toujours positif")
        else:
            cash_zero_month = int(cash_zero_idx) + 1
            st.metric("📅 Cash Zero", f"M{cash_zero_month}", 
                     delta=f"Dans {cash_zero_month - df_base['Mois'].iloc[0]} mois")
    
    # Visualisation comparative
    st.markdown("### 📈 Visualisation comparative des KPI")
    
    fig_comparative = go.Figure()
    
    # Ajouter les différentes courbes
    fig_comparative.add_trace(go.Scatter(
        x=df_base['Mois'],
        y=df_base['Cash'],
        name='Trésorerie',
        line=dict(color='green', width=3)
    ))
    
    fig_comparative.add_trace(go.Scatter(
        x=df_base['Mois'],
        y=df_base['Burn_Rate'],
        name='Burn Rate',
        yaxis='y2',
        line=dict(color='red', width=2),
        opacity=0.7
    ))
    
    fig_comparative.add_trace(go.Scatter(
        x=df_base['Mois'],
        y=df_base['Cumul_CA'] - df_base['Cumul_Couts'],
        name='Marge Cumulée',
        line=dict(color='blue', width=2, dash='dot'),
        opacity=0.7
    ))
    
    # Ajouter les lignes de référence
    fig_comparative.add_hline(y=0, line_dash="dash", line_color="gray", 
                             annotation_text="Cash = 0", annotation_position="bottom right")
    
    # Marquer le break-even
    if not pd.isna(break_even_idx):
        be_month = break_even_idx + 1
        be_value = df_base.loc[break_even_idx, 'Cumul_CA'] - df_base.loc[break_even_idx, 'Cumul_Couts']
        fig_comparative.add_vline(x=be_month, line_dash="dot", line_color="blue",
                                 annotation_text=f"Break-even M{int(be_month)}")
    
    fig_comparative.update_layout(
        title='Comparaison Trésorerie vs Burn Rate vs Marge Cumulée',
        xaxis_title='Mois',
        yaxis=dict(title='Trésorerie / Marge Cumulée (€)'),
        yaxis2=dict(
            title='Burn Rate (€/mois)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_comparative, use_container_width=True)
    
    # Graphique 2: Runway Analysis
    st.markdown("### ⏳ Analyse du Runway (Months of Runway)")
    
    # Calculer le runway à chaque mois
    df_base['Monthly_Runway'] = df_base['Cash'] / df_base['Burn_Rate'].rolling(3, min_periods=1).mean()
    df_base['Monthly_Runway'] = df_base['Monthly_Runway'].apply(lambda x: min(x, 60) if x > 0 else 0)  # Limiter à 60 mois pour lisibilité
    
    fig_runway = go.Figure()
    
    fig_runway.add_trace(go.Scatter(
        x=df_base['Mois'],
        y=df_base['Monthly_Runway'],
        name='Months of Runway',
        fill='tozeroy',
        line=dict(color='orange', width=2),
        fillcolor='rgba(255,165,0,0.2)'
    ))
    
    # Zones de danger
    fig_runway.add_hrect(y0=0, y1=3, line_width=0, fillcolor="red", opacity=0.2,
                        annotation_text="Danger", annotation_position="top left")
    fig_runway.add_hrect(y0=3, y1=6, line_width=0, fillcolor="yellow", opacity=0.2,
                        annotation_text="Attention", annotation_position="top left")
    fig_runway.add_hrect(y0=6, y1=12, line_width=0, fillcolor="lightgreen", opacity=0.2,
                        annotation_text="Confortable", annotation_position="top left")
    
    fig_runway.update_layout(
        title='Months of Runway (sur base du burn rate moyen glissant 3 mois)',
        xaxis_title='Mois',
        yaxis_title='Months of Runway',
        hovermode='x'
    )
    
    st.plotly_chart(fig_runway, use_container_width=True)
    
    # Graphique 3: Cumulative Burn vs Cumulative Revenue
    st.markdown("### 💰 Burn Cumulé vs Revenus Cumulés")
    
    df_base['Cumulative_Revenue'] = df_base['CA_total'].cumsum()
    df_base['Cumulative_Costs'] = df_base['Total_costs'].cumsum()
    df_base['Cumulative_Burn'] = df_base['Burn_Rate'].cumsum()
    
    fig_cumulative = go.Figure()
    
    fig_cumulative.add_trace(go.Scatter(
        x=df_base['Mois'],
        y=df_base['Cumulative_Revenue'],
        name='Revenus Cumulés',
        line=dict(color='green', width=3)
    ))
    
    fig_cumulative.add_trace(go.Scatter(
        x=df_base['Mois'],
        y=df_base['Cumulative_Costs'],
        name='Coûts Cumulés',
        line=dict(color='red', width=3)
    ))
    
    fig_cumulative.add_trace(go.Scatter(
        x=df_base['Mois'],
        y=df_base['Cumulative_Burn'],
        name='Burn Cumulé',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # Trouver le point de break-even
    if not pd.isna(break_even_idx):
        break_even_month = int(break_even_idx) + 1
        break_even_rev = df_base.loc[break_even_idx, 'Cumulative_Revenue']
        fig_cumulative.add_vline(x=break_even_month, line_dash="dash", line_color="blue",
                                annotation_text=f"Break-even: M{break_even_month}")
    
    fig_cumulative.update_layout(
        title='Évolution Cumulée: Revenus vs Coûts vs Burn',
        xaxis_title='Mois',
        yaxis_title='Montant Cumulé (€)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Tableau détaillé du burn
    st.markdown("### 📊 Tableau détaillé du Burn Rate")
    
    burn_summary = df_base[['Mois', 'Année', 'CA_total', 'Total_costs', 'Cash_flow', 'Burn_Rate', 'Cash']].copy()
    burn_summary['Runway_Months'] = burn_summary['Cash'] / burn_summary['Burn_Rate'].rolling(3, min_periods=1).mean()
    
    st.dataframe(
        burn_summary.style.format({
            'CA_total': '{:,.0f}',
            'Total_costs': '{:,.0f}',
            'Cash_flow': '{:,.0f}',
            'Burn_Rate': '{:,.0f}',
            'Cash': '{:,.0f}',
            'Runway_Months': '{:.1f}'
        }).applymap(
            lambda x: 'background-color: #ffcccc' if x < 0 and isinstance(x, (int, float)) else '',
            subset=['Cash_flow', 'Burn_Rate']
        ).applymap(
            lambda x: 'background-color: #ff9999' if x < 3 and isinstance(x, (int, float)) else '',
            subset=['Runway_Months']
        )
    )
    
    # Recommandations basées sur l'analyse
    st.markdown("### 🎯 Recommandations basées sur l'analyse")
    
    avg_runway = burn_summary['Runway_Months'].mean()
    
    if avg_runway < 3:
        st.error("**CRITIQUE:** Runway moyen < 3 mois. Actions immédiates nécessaires:")
        st.write("1. Réduire drastiquement les coûts fixes")
        st.write("2. Augmenter les prix ou réduire les coûts variables")
        st.write("3. Préparer une levée d'urgence")
    elif avg_runway < 6:
        st.warning("**ATTENTION:** Runway moyen < 6 mois. Actions recommandées:")
        st.write("1. Optimiser le marketing pour réduire le CAC")
        st.write("2. Revoir la structure des coûts")
        st.write("3. Planifier une levée dans les 3 mois")
    elif avg_runway < 12:
        st.info("**STABLE:** Runway moyen < 12 mois. Bonne position pour:")
        st.write("1. Poursuivre la croissance organique")
        st.write("2. Planifier une levée stratégique")
        st.write("3. Investir dans des initiatives à long terme")
    else:
        st.success("**EXCELLENT:** Runway > 12 mois. Vous pouvez:")
        st.write("1. Focus sur croissance agressive")
        st.write("2. Investir en R&D")
        st.write("3. Préparer un scale-up")

# ---------------------------------------------------------
# TAB 11 — BENCHMARKS
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
# TAB 12 — RAW DATA & EXPORT
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
    
    # Export de la cap table dynamique
    if 'cap_table_dynamic' in locals():
        csv_cap = cap_table_dynamic.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger la cap table dynamique en CSV",
            data=csv_cap,
            file_name="zolya_cap_table_dynamique.csv",
            mime="text/csv",
        )
    
    # Export de l'analyse burn
    csv_burn = burn_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger l'analyse burn rate en CSV",
        data=csv_burn,
        file_name="zolya_burn_analysis.csv",
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
