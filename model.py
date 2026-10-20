import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="Prédiction Prix Immobilier", layout="wide")

# Custom CSS pour le design
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        display: block;
        margin: 20px auto;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stNumberInput {
        background-color: white;
        border: 1.5px solid #ddd;
        border-radius: 5px;
        padding: 5px 20px;
        margin-bottom: 15px;
    }
    .stSelectbox div[data-baseweb="select"]>div, .stNumberInput input {
        text-align: left;
    }
    h1, h2, h3 {
        color: #333;
        text-align: center;
    }
    .stMarkdown {
        text-align: center;
    }
    label, .stSelectbox label, .stNumberInput label {
        font-weight: bold !important;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)


# Titre et description
st.title("🏡 Estimateur de Prix Immobiliers")
st.markdown("Entrez les caractéristiques du bien pour obtenir une estimation de son prix.")

# Chemins des fichiers
MODEL_FILENAME = 'model_gb.pkl'
SCALER_FILENAME = 'scaler.sav'
ORDINAL_ENCODER_FILENAME = 'ordinal_encoder.sav'
TARGET_ENCODER_FILENAME = 'target_encoder.sav'

# Charger le modèle
try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    st.error(f"Erreur : Fichier '{MODEL_FILENAME}' manquant.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {str(e)}")
    st.stop()

# Charger le scaler et les colonnes
try:
    scaler_tuple = joblib.load(SCALER_FILENAME)
    scaler, saved_columns = scaler_tuple
except FileNotFoundError:
    st.error(f"Erreur : Fichier '{SCALER_FILENAME}' manquant.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement du scaler : {str(e)}")
    st.stop()

try:
    ordinal_encoder = joblib.load(ORDINAL_ENCODER_FILENAME)
except FileNotFoundError:
    st.error(f"Erreur : Fichier '{ORDINAL_ENCODER_FILENAME}' manquant.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement de l'encodeur ordinal : {str(e)}")
    st.stop()

try:
    target_encoder = joblib.load(TARGET_ENCODER_FILENAME)
except FileNotFoundError:
    target_encoder = None
except Exception as e:
    st.error(f"Erreur lors du chargement de l'encodeur cible : {str(e)}")
    st.stop()

# Définir les colonnes
categorical_columns = ['Suburb', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname']
numerical_columns = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car_Park', 
                    'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']

# Colonnes attendues par le modèle
expected_columns = model.feature_names_in_

# Mappings pour Type et Method
type_display_to_model = {
    'House': 'h', 'Cottage': 'h', 'Villa': 'h', 'Semi': 'h', 'Terrace': 'h',
    'Unit': 'u', 'Duplex': 'u', 'Townhouse': 't'
}
type_display_options = list(type_display_to_model.keys())

method_display_to_model = {
    'Vendu': 'S', 'Vendu avant l\'enchère': 'SP', 'Non vendu (passé)': 'PI',
    'Offre du vendeur': 'VB', 'Vendu après l\'enchère': 'SA'
}
method_display_options = list(method_display_to_model.keys())

# Options pour les menus déroulants
options_map = {
    'Type': type_display_options,
    'Method': method_display_options,
    'Regionname': ['Northern Metropolitan', 'Western Metropolitan', 'Southern Metropolitan',
                   'Eastern Metropolitan', 'South-Eastern Metropolitan', 'Eastern Victoria',
                   'Northern Victoria', 'Western Victoria'],
    'SellerG': ['Jellis', 'Biggin', 'Rounds', 'Nelson', 'Collins', 'Dingle', 'Purplebricks', 'Greg', 'Philip', 'LITTLE', 'Kay', 'Beller', 'Marshall', 'hockingstuart', 'Brad', 'Maddison', 'Barry', 'Airport', 'Considine', 'Rendina', 'Ray', 'Propertyau', 'McDonald', 'Frank', 'Prof.', 'Harcourts', 
                'Thomson', 'Buxton', 'RT', "Sotheby's", 'Cayzer', 'Chisholm', 'McGrath', 'Brace', 'Miles', 'Holland', 'Love', 'Thomas', 'Barlow', 'Sweeney', 'Village', 'Jas', 'Williams', 'Gunn&Co', 'Burnham', 'Point', 'Compton', 'FN', 'Hunter', 'Raine', 'Noel', 'Hodges', 'Gary', 'Fletchers', 'Woodards', 'Raine', 'Alexkarbon', 'Walshe', 'Weda', 'Stockdale', 
                'Tim', 'Buxton/Marshall', 'AIME', 'Moonee', 'HAR', 'Edward', 'RW', 'North', 'MICM', 'Ascend', 'Christopher', 'Morrison', 'Bekdon', 'Mandy', 'R&H', 'Fletchers/One', 'Lindellas', 'Appleby', 'Assisi', 'One', 'Owen', 'Parkes', 'C21', 'Century', "O'Brien", 'Bayside', 'Anderson', 'Paul', 'hockingstuart/Buxton', 'Smart', 'First', 'Matthew', 'Melbourne', 'Nick', 'Allens', 'Bells', 
                'Nguyen', 'Trimson', 'Douglas', 'Batty', 'GL', 'YPA', "Tiernan's", 'J', 'Castran', 'Branon', 'Rodney', 'Joseph', 'Harrington', 'Galldon', 'Grantham', 'Chambers', 'Pagan', 'Peter', 'hockingstuart/Advantage', 'Ross', 'Rayner', 'Garvey', "O'Donoghues", 'Hamilton', 'Weast', 'Kelly', 'Property', "Private/Tiernan's", 'Australian', 'Bustin', "Abercromby's", 'Whiting', 'iTRAK', 'Del', 'Caine', 'Nicholson', 'JMRE', 'ASL',
                'Changing', 'Re', 'RE', 'Walsh', 'Foxtons', 'Ham', 'Darren', 'Vic', 'Haughton', 'Scott', 'Pride', 'Morleys', 'Wilson', 'Buxton/Advantage', 'Professionals', 'Joe', 'Red', 'hockingstuart/Jellis', 'Craig', 'Naison', 'Sweeney/Advantage', 'Reach', 'Domain', 'Eview', 'Jason', "D'Aprano", 'Charlton', 'CASTRAN', 'Wood', 'William', 'Coventry', 'Buckingham', 'Nardella', 'Karen', 'LJ', 'Shape', 'Direct',
                'Fletchers/Fletchers', 'Besser', 'Johnston', 'Avion', 'Redina', 'Clairmont', 'Lucas', 'Dixon', 'Xynergy', 'Elite', 'Buxton/Find', 'W.B.', 'Leased', 'Harcourts/Barry', 'Win', 'Collings', 'New', 'Parkinson', 'Geoff', 'hockingstuart/Barry', 'Blue', 'MSM', 'Steveway', 'hockingstuart/Village', 'VICPROP', 'Inner', 'Meadows', 'Allan', 'S&L', 'Crane', 'Calder', 'Metro', 'Homes', 'Zahn', 'Mason', 'Landfield', 'David', 'Luxe', 'Prowse', 'Ken', 'Rombotis', 'iOne', 'hockingstuart/Sweeney', 
                'Iconek', 'Luxton', 'Hooper', 'JRW', 'Oak', 'White', 'Jim', 'Weston', '@Realty', 'Reed', 'Oriental', 'Max', 'Steller', 'Rosin', 'Real', 'Jeffrey', 'Hall', 'Icon', 'buyMyplace', 'WHITEFOX', 'Christou', 'Marvelli', "Grant's", 'Hoskins', 'Gardiner', 'McLennan', 'Triwest', 'Reliance', 'PRDNationwide', 'Only', 'Leyton', 'iSell', 'Obrien', 'Maitland', 'Millership', 'L', 'Emerson', 'Follett', 'Ash', 'LLC', 'Ace', 'Flannagan', 'Carter', 'WeSell', 'M.J', 'iProperty', 'Hayeswinckle', 
                'Schroeder', 'iHomes', 'VICProp', 'REMAX', 'Space', 'Victory', 'Aumeca', 'Mindacom', 'Ryder', 'Sanctuary', 'Leeburn', 'Asset', 'Westside', 'Sell', 'Mitchell', 'Darras', 'U', 'Nicholas', 'Rexhepi', 'Keatings', 'Aquire', 'Conquest', 'Prime', 'Donovan', 'Community', 'Free', 'Veitch', 'John', 'Peake', 'Ristic', 'JY', 'Upper', 'Daniel', 'VicHomes', 'TRUE', 'McNaughton', 'Leading', 'Bullen', 'Better', 'Boutique', 'Royston', 'Langwell', 'Silver', 'Wyndham', 'Kaye', 'Nicholls', 'Bowman',
                'Just', 'Home', 'Le', 'Cooper', 'Black', 'The', 'Open', 'Roger', 'McEwing', 'Sandhurst', 'Boran', 'hockingstuart/hockingstuart', 'Area', 'MJ', 'Eric', 'SN', 'Waterfront', 'P', 'PRD', 'Bombay', 'Quinta', 'T', 'Tanner', 'LJH', 'Munn', 'Knight', 'Justin', 'Blackbird', 'Sweeney/Burnham', 'H', 'Alex', 'Can', 'Pavilion', 'Benchmark', 'Gellibrand', 'hockingstuart/Harcourts', 'Charter', 'Surreal', 'Vicprop', 'A', 'Upside', 'OBrien', 'Meallin', 'Malvern', 'Mark', 'Bradly', 'For', 'Peninsula',
                'Sutherland', 'Leaders', 'Spencer', 'McGrath/Langwell', 'Benlor', 'Burns', 'Sterling', 'K.R.Peters', 'Buxton/Buxton', 'CarlinSmith', 'Gold', 'Hill', 'People', 'hockingstuart/Biggin', 'Mega', 'Skad', 'Ruralco', 'Create', 'Janice', 'R', 'G&H', 'Commercial', 'PSP', 'hockingstuart/Cayzer', 'Unity', 'Sprint', 'McGrath/Buxton', 'hockingstuart/Marshall', 'McGrath/First', 'voglwalpole', 'Watermark', 'Methven', 'Unknown'],
    'CouncilArea': ['Yarra City Council', 'Moonee Valley City Council', 'Port Phillip City Council', 'Darebin City Council', 'Hobsons Bay City Council', 'Stonnington City Council', 'Boroondara City Council', 'Monash City Council', 'Glen Eira City Council', 'Whitehorse City Council', 'Maribyrnong City Council', 'Bayside City Council', 'Moreland City Council', 'Manningham City Council', 'Melbourne City Council', 'Banyule City Council', 'Brimbank City Council', 'Kingston City Council', 'Hume City Council', 'Knox City Council', 'Maroondah City Council', 'Casey City Council', 'Melton City Council',
                    'Greater Dandenong City Council', 'Nillumbik Shire Council', 'Cardinia Shire Council', 'Whittlesea City Council', 'Frankston City Council', 'Macedon Ranges Shire Council', 'Yarra Ranges Shire Council', 'Wyndham City Council', 'Moorabool Shire Council', 'Mitchell Shire Council', 'Unknown'],
    'Suburb': ['Abbotsford', 'Airport West', 'Albert Park', 'Alphington', 'Altona', 'Altona North', 'Armadale', 'Ascot Vale', 'Ashburton', 'Ashwood', 'Avondale Heights', 'Balaclava', 'Balwyn', 'Balwyn North', 'Bentleigh', 'Bentleigh East', 'Box Hill', 'Braybrook', 'Brighton', 'Brighton East', 'Brunswick', 'Brunswick West', 'Bulleen', 'Burwood', 'Camberwell', 'Canterbury', 'Carlton North', 'Carnegie', 'Caulfield', 'Caulfield North', 'Caulfield South', 'Chadstone', 'Clifton Hill', 'Coburg', 'Coburg North', 'Collingwood', 'Doncaster', 'Eaglemont', 'Elsternwick', 'Elwood', 'Essendon', 'Essendon North', 'Fairfield', 'Fitzroy', 'Fitzroy North', 'Flemington', 'Footscray', 'Glen Iris', 'Glenroy', 'Gowanbrae', 'Hadfield', 'Hampton', 'Hampton East', 'Hawthorn', 'Heidelberg Heights', 'Heidelberg West', 'Hughesdale', 'Ivanhoe', 'Kealba', 'Keilor East', 'Kensington', 'Kew', 'Kew East', 'Kooyong', 'Maidstone', 'Malvern', 'Malvern East', 'Maribyrnong', 'Melbourne', 'Middle Park', 'Mont Albert', 'Moonee Ponds', 'Moorabbin', 'Newport', 'Niddrie', 'North Melbourne', 'Northcote', 'Oak Park', 'Oakleigh South', 'Parkville', 'Pascoe Vale', 'Port Melbourne', 'Prahran', 'Preston', 'Reservoir', 'Richmond', 'Rosanna', 'Seddon', 'South Melbourne', 'South Yarra', 'Southbank', 'Spotswood', 'St Kilda', 'Strathmore', 'Sunshine', 'Sunshine North', 'Sunshine West', 'Surrey Hills', 'Templestowe Lower', 'Thornbury', 'Toorak', 'Viewbank', 'Watsonia', 'West Melbourne', 'Williamstown', 'Williamstown North', 'Windsor', 'Yallambie', 'Yarraville', 'Aberfeldie', 'Bellfield', 'Brunswick East', 'Burnley', 'Campbellfield', 'Carlton', 'East Melbourne', 'Essendon West', 'Fawkner', 'Hawthorn East', 'Heidelberg', 'Ivanhoe East', 'Jacana', 'Kingsbury', 'Kingsville', 'Murrumbeena', 'Ormond', 'West Footscray', 'Albion', 'Brooklyn', 'Glen Huntly', 'Oakleigh', 'Ripponlea', 'Cremorne', 'Docklands', 'South Kingsville', 'Strathmore Heights', 'Travancore', 'Caulfield East', 'Seaholme', 'Keilor Park', 'Gardenvale', 'Princes Hill', 'Altona Meadows', 'Ardeer', 'Attwood', 'Baysmater', 'Bayswater North', 'Beaumaris', 'Berwick', 'Blackburn', 'Blackburn South', 'Boronia', 'Briar Hill', 'Broadmeadows', 'Bundoora', 'Burnside Heights', 'Burwood East', 'Cairnlea', 'Caroline Springs', 'Cheltenham', 'Clarinda', 'Clayton', 'Clyde North', 'Craigieburn', 'Cranbourne', 'Croydon', 'Croydon Hills', 'Croydon North', 'Dandenong', 'Dandenong North', 'Diamond Creek', 'Dingley Village', 'Doncaster East', 'Donvale', 'Doreen', 'Edithvale', 'Eltham', 'Eltham North', 'Emerald', 'Epping', 'Eumemmerring', 'Ferntree Gully', 'Forest Hill', 'Frankston', 'Frankston North', 'Frankston South', 'Gisborne', 'Gladstone Park', 'Glen Waverley', 'Greensborough', 'Greenvale', 'Hallam', 'Healesville', 'Heathmont', 'Highett', 'Hillside', 'Hoppers Crossing', 'Huntingdale', 'Keilor Downs', 'Keilor Lodge', 
                    'Keysborough', 'Kings Park', 'Lalor', 'Lower Plenty', 'Lynbrook', 'MacLeod', 'Melton', 'Melton South', 'Melton West', 'Mentone', 'Mernda', 'Mickleham', 'Mill Park', 'Mitcham', 'Montmorency', 'Montrose', 'Mordialloc', 'Mount Waverley', 'Mulgrave', 'Narre Warren', 'Noble Park', 'Nunawading', 'Oakleigh East', 'Parkdale', 'Point Cook', 'Ringwood', 'Ringwood East', 'Rockbank', 'Rowville', 'Roxburgh Park', 'Sandringham', 'Seabrook', 'Seaford', 'Skye', 'Springvale', 'Springvale South', 'St Albans', 'St Helena', 'Sunbury', 'Sydenham', 'Tarneit', 'Taylors Hill', 'Taylors Lakes', 'Tecoma', 'Templestowe', 'The Basin', 'Thomastown', 'Truganina', 'Tullamarine', 'Vermont', 'Vermont South', 'Wantirna', 'Wantirna South', 'Werribee', 'Werribee South', 'Westmeadows', 'Williams Landing', 'Wollert', 'Wyndham Vale', 'Aspendale', 'Black Rock', 'Blackburn North', 'Bonbeach', 'Carrum', 'Chelsea', 'Clayton South', 'Dallas', 'Delahey', 'Doveton', 'McKinnon', 'Mooroolbark',
                    'Pakenham', 'Ringwood North', 'Scoresby', 'Warrandyte', 'Watsonia North', 'Wattle Glen', 'Wheelers Hill', 'Albanvale', 'Aspendale Gardens', 'Belgrave', 'Carrum Downs', 'Cranbourne East', 'Cranbourne North', 'Deer Park', 'Heatherton', 'Kilsyth', 'Langwarrin', 'Notting Hill', 'Patterson Lakes', 'Riddells Creek', 'Beaconsfield Upper', 'Chelsea Heights', 'Chirnside Park', 'Coolaroo', 'Darley', 'Hampton Park', 'Keilor', 'Meadow Heights', 'Mount Evelyn', 'North Warrandyte', 'Sandhurst', 'Silvan', 'Wallan', 'Croydon South', 'Derrimut', 'Diggers Rest', 'Knoxfield', 'Upwey', 'Warranwood', 'Bacchus Marsh', 'Bullengarook', 'Deepdene', 'Hurstbridge', 'Kurunjang', 'Laverton', 'Lilydale', 'Wonga Park', 'Endeavour Hills', 'Officer', 'Olinda', 'Waterways', 'Beaconsfield', 'Yarra Glen', 'Brookfield', 'Lysterfield', 'Plenty', 'Whittlesea', 'Burnside', 'New Gisborne', 'Guys Hill', 'Plumpton', 'croydon', 'Monbulk', 'Avonsleigh', 'Wildwood', 'Gisborne South', 'Research', 'viewbank', 'Botanic Ridge', 'Bulla', 'Coldstream', 'Hopetoun Park', 'Cranbourne West', 'Eynesbury', 'Fawkner Lot', 'Ferny Creek', 'Wandin North', 'Kalkallo', 'Menzies Creek', 'Unknown'],
    'Date': ['3/12/2016', '4/02/2016', '4/03/2017', '4/06/2016', '7/05/2016',
       '8/10/2016', '10/12/2016', '12/11/2016', '13/08/2016',
       '14/05/2016', '15/10/2016', '16/07/2016', '17/09/2016',
       '18/03/2017', '18/06/2016', '19/11/2016', '22/05/2016',
       '22/08/2016', '23/04/2016', '24/09/2016', '25/02/2017',
       '26/07/2016', '27/11/2016', '28/05/2016', '30/07/2016',
       '3/09/2016', '6/08/2016', '7/11/2016', '10/09/2016', '11/02/2017',
       '16/04/2016', '27/06/2016', '28/08/2016', '12/06/2016',
       '11/03/2017', '28/01/2016', '6/05/2017', '8/04/2017', '13/05/2017',
       '29/04/2017', '20/05/2017', '22/04/2017', '1/07/2017', '3/06/2017',
       '17/06/2017', '24/06/2017', '27/05/2017', '8/07/2017',
       '12/08/2017', '15/07/2017', '22/07/2017', '29/07/2017',
       '3/09/2017', '9/09/2017', '16/09/2017', '19/08/2017', '23/09/2017',
       '26/08/2017', '7/10/2017', '14/10/2017', '21/10/2017',
       '28/10/2017', '30/09/2017', '4/11/2017', '11/11/2017',
       '18/11/2017', '25/11/2017', '6/01/2018', '9/12/2017', '3/02/2018',
       '10/02/2018', '20/01/2018', '27/01/2018', '3/03/2018',
       '10/03/2018', '17/02/2018', '17/03/2018', '24/02/2018']
}

# Descriptions conviviales pour les labels
label_map = {
    'Suburb': 'Le quartier ou la zone où se trouve le bien immobilier',
    'Type': 'Le type de bien immobilier',
    'Method': 'La méthode de vente',
    'SellerG': 'Le nom de l’agence ou du vendeur',
    'Date': 'Date de vente',
    'CouncilArea': 'Administration locale responsable de la gestion de la zone où se situe le bien immobilier',
    'Regionname': 'La région générale',
    'Rooms': 'Nombre de pièces',
    'Distance': 'Distance depuis le centre-ville (km)',
    'Postcode': 'Code postal',
    'Bedroom2': 'Nombre de chambres (secondaire)',
    'Bathroom': 'Nombre de salles de bain',
    'Car_Park': 'Nombre de places de stationnement',
    'Landsize': 'Surface du terrain (m²)',
    'BuildingArea': 'Surface du bâtiment (m²)',
    'YearBuilt': 'Année de construction',
    'Lattitude': 'Latitude',
    'Longtitude': 'Longitude',
    'Propertycount': 'Nombre de propriétés dans le quartier'
}

# Fonction pour enrichir les caractéristiques
def enrich_features(df, training_columns=None):
    df = df.copy()
    
   # Assurer des valeurs minimales pour éviter des ratios extrêmes
    df['Bathroom'] = df['Bathroom'].clip(lower=1)
    df['Bedroom2'] = df['Bedroom2'].clip(lower=1)
    df['Landsize'] = df['Landsize'].clip(lower=50)
    df['BuildingArea'] = df['BuildingArea'].clip(lower=20)
    
    # Aligner avec les colonnes d'entraînement
    if training_columns is not None:
        for col in training_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[training_columns]
    
    return df

# Fonction pour capper les outliers
def cap_outliers(df, col, lower_percentile=0.05, upper_percentile=0.95):
    lower_limit = df[col].quantile(lower_percentile)
    upper_limit = df[col].quantile(upper_percentile)
    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
    return df

# Fonction pour encoder les variables catégoriques
def smart_numeric_encoding(df, ordinal_cols, targetenc_cols, ordinal_encoder, target_encoder):
    df_encoded = df.copy()
    if ordinal_cols:
        try:
            valid_ordinal_cols = [col for col in ordinal_cols if col in ordinal_encoder.feature_names_in_]
            if not valid_ordinal_cols:
                st.error("Erreur : Aucune colonne ordinale valide trouvée.")
                st.stop()
            df_encoded[valid_ordinal_cols] = ordinal_encoder.transform(df_encoded[valid_ordinal_cols])
        except Exception as e:
            st.error(f"Erreur dans l'encodage ordinal : {str(e)}")
            st.stop()
    if targetenc_cols and target_encoder is not None:
        try:
            valid_targetenc_cols = [col for col in targetenc_cols if col in target_encoder.cols]
            if not valid_targetenc_cols:
                st.error("Erreur : Aucune colonne valide pour le target encoding.")
                st.stop()
            df_encoded[valid_targetenc_cols] = target_encoder.transform(df_encoded[valid_targetenc_cols])
        except Exception as e:
            st.error(f"Erreur dans le target encoding : {str(e)}")
            st.stop()
    return df_encoded

# Interface utilisateur
st.header("Caractéristiques du Bien")
input_data = {}

# Vérification des valeurs inconnues
for col in categorical_columns:
    if col in options_map:
        unknown_values = set(options_map[col]) - set(options_map[col])
        if unknown_values:
            st.warning(f"Valeurs inconnues dans {col} : {unknown_values}")

# Organiser les entrées en 3 colonnes
all_inputs = categorical_columns + ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car_Park', 
                                    'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']
rows = [all_inputs[i:i+3] for i in range(0, len(all_inputs), 3)]

for row in rows:
    cols = st.columns(3)
    for i, col_name in enumerate(row):
        with cols[i]:
            if col_name in categorical_columns:
                input_data[col_name] = st.selectbox(
                    f"{col_name} ({label_map.get(col_name, col_name)})",
                    options_map[col_name],
                    key=col_name,
                    help=label_map.get(col_name, "Aucune description")
                )
            else:
                if col_name == 'Rooms':
                    input_data[col_name] = st.number_input(
                        'Nombre de pièces', min_value=1, max_value=10, value=3, key=col_name
                    )
                elif col_name == 'Bedroom2':
                    max_bedroom2 = input_data.get('Rooms', 10)
                    input_data[col_name] = st.number_input(
                        'Nombre de chambres (secondaire)', min_value=0, max_value=max_bedroom2, value=min(3, max_bedroom2), key=col_name
                    )
                elif col_name == 'Distance':
                    input_data[col_name] = st.number_input('Distance depuis le centre-ville (km)', min_value=0.0, max_value=50.0, value=10.0, key=col_name)
                elif col_name == 'Postcode':
                    input_data[col_name] = st.number_input('Code postal', min_value=3000, max_value=4000, value=3000, key=col_name)
                elif col_name == 'Bathroom':
                    input_data[col_name] = st.number_input('Nombre de salles de bain', min_value=1, max_value=5, value=1, key=col_name)
                elif col_name == 'Car_Park':
                    input_data[col_name] = st.number_input('Nombre de places de stationnement', min_value=0, max_value=5, value=1, key=col_name)
                elif col_name == 'Landsize':
                    input_data[col_name] = st.number_input('Surface du terrain (m²)', min_value=50.0, max_value=5000.0, value=500.0, key=col_name)
                elif col_name == 'BuildingArea':
                    max_building_area = input_data.get('Landsize', 5000.0)
                    input_data[col_name] = st.number_input('Surface du bâtiment (m²)', min_value=20.0, max_value=max_building_area, value=min(150.0, max_building_area), key=col_name)
                elif col_name == 'YearBuilt':
                    input_data[col_name] = st.number_input('Année de construction', min_value=1850, max_value=datetime.now().year, value=2000, key=col_name)
                elif col_name == 'Lattitude':
                    input_data[col_name] = st.number_input('Latitude', min_value=-38.0, max_value=-37.0, value=-37.8, key=col_name)
                elif col_name == 'Longtitude':
                    input_data[col_name] = st.number_input('Longitude', min_value=144.0, max_value=145.5, value=144.9, key=col_name)
                elif col_name == 'Propertycount':
                    input_data[col_name] = st.number_input('Nombre de propriétés dans le quartier', min_value=0, max_value=20000, value=5000, key=col_name)

# Validation des entrées
if input_data.get('Bedroom2', 0) > input_data.get('Rooms', 10):
    st.error("Le nombre de chambres ne peut pas dépasser le nombre de pièces.")
    st.stop()
if input_data.get('BuildingArea', 0) > input_data.get('Landsize', 5000):
    st.error("La surface du bâtiment ne peut pas dépasser la surface du terrain.")
    st.stop()

# Bouton pour lancer la prédiction
if st.button("Estimer le Prix"):
    input_df = pd.DataFrame([input_data])

    # Convertir les valeurs conviviales de Type et Method en valeurs du modèle
    input_df['Type'] = input_df['Type'].map(type_display_to_model)
    input_df['Method'] = input_df['Method'].map(method_display_to_model)

    # Capping des outliers
    cols_to_cap = ['Landsize', 'BuildingArea', 'Distance', 'Bathroom', 'Bedroom2', 'Car_Park']
    for col in cols_to_cap:
        input_df = cap_outliers(input_df, col)

    # Encoder les colonnes catégoriques
    ordinal_cols = ['Type', 'Method', 'Regionname']
    targetenc_cols = ['Suburb', 'SellerG', 'Date', 'CouncilArea']
    input_df = smart_numeric_encoding(input_df, ordinal_cols, targetenc_cols, ordinal_encoder, target_encoder)

    # Enrichir les caractéristiques
    input_df = enrich_features(input_df, training_columns=expected_columns)

    # Vérifier les colonnes avant standardisation
    num_cols = [col for col in saved_columns if col in input_df.columns]
    missing_cols = [col for col in saved_columns if col not in input_df.columns]
    if missing_cols:
        st.warning(f"Colonnes manquantes : {missing_cols}")

    # Standardisation
    try:
        input_df[num_cols] = scaler.transform(input_df[num_cols])
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {str(e)}")
        st.stop()

    # Réordonner les colonnes
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Prédiction
    try:
        predicted_price = model.predict(input_df)
        st.markdown("""
            <div style='
                background-color:#e8f5e9;
                padding:30px;
                border-radius:15px;
                text-align:center;
                font-size:40px;
                font-weight:bold;
                color:#1b5e20;
                margin-top:30px;
                box-shadow:0 4px 10px rgba(0,0,0,0.1);'>
                💰 Prix estimé : ${:,.2f}
            </div>
        """.format(predicted_price[0]), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")