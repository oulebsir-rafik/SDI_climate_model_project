🔹 Bassin de captage (BC)
    • pH_BC : pH de l’eau au bassin de captage 
    • Température_BC : température de l’eau au bassin de captage 
    • Conductivité_BC : conductivité électrique de l’eau au bassin de captage 
    • TDS_BC : concentration en solides dissous totaux au bassin de captage 
    • Turbidité_BC : turbidité de l’eau au bassin de captage 

🔹 Sortie Décantation ligne A (O_DE_A)
    • Ph_O_DE_A : pH en sortie de la décantation ligne A 
    • Température_O_DE_A : température en sortie de la décantation ligne A 
    • Conductivité_O_DE_A : conductivité en sortie de la décantation ligne A 
    • TDS_O_DE_A : TDS en sortie de la décantation ligne A 
    • Turbidité_O_DE_A : turbidité en sortie de la décantation ligne A 

🔹 Sortie Décantation ligne B (O_DE_B)
    • Ph_O_DE_B : pH en sortie de la décantation ligne B 
    • Température_O_DE_B : température en sortie de la décantation ligne B 
    • Conductivité_O_DE_B : conductivité en sortie de la décantation ligne B 
    • TDS_O_DE_B : TDS en sortie de la décantation ligne B 
    • Turbidité_O_DE_B : turbidité en sortie de la décantation ligne B 

🔹 Sortie Filtre à sable ligne A (O_SF_A)
    • Ph_O_SF_A : pH en sortie du filtre à sable ligne A 
    • Température_O_SF_A : température en sortie du filtre à sable ligne A 
    • Conductivité_O_SF_A : conductivité en sortie du filtre à sable ligne A 
    • TDS_O_SF_A : TDS en sortie du filtre à sable ligne A 
    • Turbidité_O_SF_A : turbidité en sortie du filtre à sable ligne A 

🔹 Sortie Filtre à sable ligne B (O_SF_B)
    • Ph_O_SF_B : pH en sortie du filtre à sable ligne B 
    • Température_O_SF_B : température en sortie du filtre à sable ligne B 
    • Conductivité_O_SF_B : conductivité en sortie du filtre à sable ligne B 
    • TDS_O_SF_B : TDS en sortie du filtre à sable ligne B 
    • Turbidité_O_SF_B : turbidité en sortie du filtre à sable ligne B 

🔹 Sortie Filtre à cartouche ligne A (O_CF_A)
    • Ph_O_CF_A : pH en sortie du filtre à cartouche ligne A 
    • Température_O_CF_A : température en sortie du filtre à cartouche ligne A 
    • Conductivité_O_CF_A : conductivité en sortie du filtre à cartouche ligne A 
    • TDS_O_CF_A : TDS en sortie du filtre à cartouche ligne A 
    • Turbidité_O_CF_A : turbidité en sortie du filtre à cartouche ligne A 

🔹 Sortie Filtre à cartouche ligne B (O_CF_B)
    • Ph_O_CF_B : pH en sortie du filtre à cartouche ligne B 
    • Température_O_CF_B : température en sortie du filtre à cartouche ligne B 
    • Conductivité_O_CF_B : conductivité en sortie du filtre à cartouche ligne B 
    • TDS_O_CF_B : TDS en sortie du filtre à cartouche ligne B 
    • Turbidité_O_CF_B : turbidité en sortie du filtre à cartouche ligne B 

🔹 Sortie Osmose inverse (O_RO)
    • Ph_O_RO : pH en sortie de l’osmose inverse 
    • Température_O_RO : température en sortie de l’osmose inverse 
    • Conductivité_O_RO : conductivité en sortie de l’osmose inverse 
    • TDS_O_RO : TDS en sortie de l’osmose inverse 
    • Turbidité_O_RO : turbidité en sortie de l’osmose inverse 

🔹 Sortie Expédition (O_EX)
    • Ph_O_EX : pH de l’eau à l’expédition 
    • Température_O_EX : température de l’eau à l’expédition 
    • Conductivité_O_EX : conductivité à l’expédition 
    • TDS_O_EX : TDS à l’expédition 
    • Chlore libre_O_EX : concentration en chlore libre à l’expédition 
    • Turbidité_O_EX : turbidité de l’eau à l’expédition
­

Ces variables proviennent des scripts d'acquisition de données géospatiales dans `src/`. Elles ne sont pas mesurées à l'usine mais téléchargées depuis des sources externes (ré-analyses climatiques, satellites, modèles océaniques) pour un point de référence géographique.

⚠️ Deux points de référence différents sont utilisés selon la variable :
- **Point de captage marin de l'usine** (lat=36.75°N, lon=3.06°E) : utilisé par `copernicus_marine_extraction.py`, car les conditions marines sont liées au point de pompage d'eau de mer, pas au bassin versant.
- **Centroïde du bassin versant Corso** délinéé par `corso_watershed.py` (`output/corso_watershed/watershed.geojson`) : utilisé par `era5_data_extraction.py` et `chirps_data_extraction.py`, car ces variables décrivent le climat/les précipitations du bassin versant amont, pas la zone marine.

🔹 Délinéation du bassin versant (`corso_watershed.py` → `report.json`, `watershed.geojson`, `outlet_snapped.geojson`)
    • outlet_input.latitude / outlet_input.longitude : coordonnées de l’exutoire fournies en entrée (WGS84)
    • outlet_final.latitude / outlet_final.longitude : coordonnées de l’exutoire après ajustement (« snapping ») sur le réseau de drainage
    • working_crs : système de projection (UTM) utilisé pour les calculs hydrologiques
    • area_km2 : superficie du bassin versant délinéé, en km²
    • snapping.max_distance_m / snapping.accumulation_threshold : paramètres de l’algorithme de snapping de l’exutoire sur le cours d’eau
    • snapping.snap_distance_m : distance réelle entre l’exutoire d’origine et l’exutoire ajusté, en mètres
    • snapping.snap_method : méthode utilisée pour l’ajustement (« threshold_stream » ou repli sur l’accumulation de flux maximale)
    • snapping.large_snap_warning : indique si la distance de snapping dépasse le seuil d’alerte
    • dem.used_source / dem.requested_source : source du modèle numérique de terrain utilisée (SRTM 30m ou Copernicus DEM 30m), et repli éventuel de l’une vers l’autre
    • validation.edge_truncation_warning : indique si le bassin touche le bord de l’emprise du MNT (risque de troncature)
    • validation.geometry_valid : validité géométrique du polygone du bassin versant

🔹 Climat ERA5 au point du bassin versant (`era5_data_extraction.py` → `data/clean/era5_daily_point_Jan2024_Feb2025.csv`)
    • date : date (agrégation journalière)
    • requested_latitude / requested_longitude : coordonnées du centroïde du bassin versant demandées
    • era5_latitude / era5_longitude : coordonnées de la maille ERA5 (0.25°) réellement la plus proche
    • t2m_mean_c / t2m_min_c / t2m_max_c : température de l’air à 2m, moyenne/min/max journalière (°C)
    • dewpoint_mean_c : température du point de rosée à 2m, moyenne journalière (°C)
    • precipitation_mm : précipitation totale journalière (mm)
    • wind_speed_mean_ms / wind_speed_max_ms : vitesse du vent à 10m (norme de u10/v10), moyenne/max journalière (m/s)
    • surface_pressure_mean_hpa : pression atmosphérique de surface, moyenne journalière (hPa)
    • solar_radiation_mj_m2 : rayonnement solaire descendant en surface, cumul journalier (MJ/m²)

🔹 Précipitations CHIRPS sur le bassin versant (`chirps_data_extraction.py`)
    Résumé zonal journalier (`data/clean/chirps_daily_watershed_Jan2024_Feb2025.csv`) :
    • date : date journalière
    • precip_mean_mm / precip_max_mm / precip_min_mm : précipitation moyenne/max/min sur les pixels CHIRPS (0.05°) intersectant le bassin versant (mm)
    • pixel_count : nombre de pixels CHIRPS intersectant le bassin versant ce jour-là

    Table diagnostique par pixel (`data/rawdata/chirps_pixels_watershed_Jan2024_Feb2025.csv`) :
    • pixel_lat / pixel_lon : coordonnées du centre du pixel
    • precipitation_mm : précipitation du pixel (mm)
    • is_max : indique si ce pixel porte la valeur maximale du jour

🔹 Océan (Copernicus Marine) au point de captage marin (`copernicus_marine_extraction.py`)
    Fichier fusionné (`data/clean/copernicus_marine_daily_Feb2024_Feb2025.csv`) et fichiers par produit (`data/clean/copernicus_marine_<Produit>_daily_Feb2024_Feb2025.csv`), avec pour chaque variable `{var}` :
    • {var}_median : valeur médiane journalière parmi les points de grille voisins retenus (agrégation robuste, préférée à la moyenne ou au point le plus proche)
    • {var}_n_points : nombre de points de grille ayant contribué à la médiane ce jour-là
    • {var}_dist_min_km / {var}_dist_median_km / {var}_dist_max_km : distances (min/médiane/max) entre le point de captage réel et les points de grille utilisés, en km — traçabilité qualité

    Produits et variables extraites :
    • Physics_Temperature → thetao : température de l’eau de mer (°C)
    • Physics_Salinity → so : salinité (PSU)
    • Physics_Currents → uo, vo : composantes est/nord du courant marin (m/s)
    • Sea_Surface_Height → zos : hauteur de la surface de la mer (m)
    • Waves → VHM0 : hauteur significative des vagues (m)
    • Wind → eastward_wind, northward_wind : composantes est/nord du vent en surface (m/s)
    • Ocean_Color_SPM_Turbidity_Regional → TUR, SPM : turbidité et matières en suspension (couleur de l’eau, satellite)
    • Biogeochemistry_Carbon_pH → ph, dissic : pH de l’eau de mer et carbone inorganique dissous
    • Biogeochemistry_Oxygen → o2 : concentration en oxygène dissous
    • Biogeochemistry_Plankton_Biomass → chl, phyc : chlorophylle-a et biomasse phytoplanctonique