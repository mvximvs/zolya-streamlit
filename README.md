# 📊 Zolya — Business Plan & Financial Simulator

Simulateur Streamlit pour projeter le business plan de **Zolya** : croissance utilisateurs, revenus, coûts, trésorerie, scénarios de croissance et valorisation (cap table).

L’app permet de tester rapidement différents setups (prix, churn, marketing, salaires, CAPEX…) et d’en déduire :
- la trajectoire d’utilisateurs,
- le chiffre d’affaires par type de produit,
- le burn mensuel et la trésorerie,
- plusieurs scénarios (Safe / Base / Moon),
- une valorisation basée sur un multiple d’ARR,
- une cap table simplifiée post-levée.

---

## ⚙️ Fonctionnalités principales

- **Modèle utilisateurs** :
  - Croissance logistique : `r · U · (1 − U/K)`
  - Acquisition marketing : `Budget marketing / CAC`
  - Churn mensuel configurable
- **Revenus** :
  - Abonnement **Basic** (€/mois)
  - Abonnement **Premium** (€/mois) + mix Basic/Premium
  - Revenus Biomarkers (prix, coût et taux d’achat annuel)
- **Coûts** :
  - Masse salariale (fondateurs + employés)
  - Loyer / bureaux
  - Outils SaaS / infra / IA
  - Autres coûts fixes
  - Marketing
  - Coûts variables Biomarkers
  - Frais de paiement (% du CA)
  - CAPEX annuel, décaissé à un mois donné
- **Trésorerie & unit economics** :
  - Cash flow mensuel et trésorerie cumulée
  - ARPU mensuel
  - LTV approximative (`ARPU / churn`)
  - Ratio LTV / CAC
- **Scénarios** :
  - `Safe` : CAC ↑, churn ↑, marketing ↓
  - `Base` : hypothèses telles que définies dans la sidebar
  - `Moon` : CAC ↓, churn ↓, marketing ↑
- **Valorisation & cap table** :
  - ARR de l’année choisie (Base)
  - Pré-money = ARR × multiple
  - Post-money = Pré-money + montant levé
  - Répartition du capital : fondateurs / investisseurs / option pool
- **Exports & benchmarks** :
  - Export CSV des projections mensuelles (Base)
  - Rappel des hypothèses en JSON
  - Tables de benchmarks de marché et de multiples (indicatif)

---

## 📦 Installation

1. Cloner le repo ou copier le fichier `app.py` :

```bash
git clone <ton-repo-ou-dossier>
cd <ton-repo>
