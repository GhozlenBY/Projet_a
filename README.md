# Projet_a : Projet de réconnaissance des gestes : "Yes (✓)" et "No (X)" avec Tensor_flow et Arduino

# Description

Ce projet implémente une application TinyML sur un kit Arduino pour reconnaître des gestes simples ("yes" et "no") en utilisant TensorFlow Lite pour Microcontrôleurs. Le programme interagit avec l'utilisateur en posant des questions et en évaluant les réponses basées sur des gestes capturés par l'accéléromètre.

# Matériel Utilisée
• Kit Arduino Tiny Machine Learning
• Câble USB pour la connexion avec le PC

# Étapes du Projet

1. Génération des Données d'Entraînement :

• Utilisation du script generate_data_to_train pour lire les données de l'accéléromètre et du gyroscope de l'Arduino.
• Les données sont imprimées dans le moniteur série en format CSV.

2. Enregistrement des Données :

• Enregistrement des données dans deux fichiers .csv (yes.csv et no.csv) en utilisant le script Python Data_Collect pour la collecte des données via la bibliothèque Serial.

3. Entraînement du Modèle :

• Utilisation du code arduino_tiny_ml avec la bibliothèque TensorFlow pour entraîner le modèle sur les données collectées.
• Génération du fichier model.h qui contient le modèle entraîné.

4. Test du Modèle sur Arduino :

• Téléversement du code du jeu Yes_No_Question_Game sur l'Arduino.
• Le programme pose des questions et utilise les gestes détectés par l'IMU pour évaluer les réponses.

# Utilisation
Lancez le programme et répondez aux questions posées par des gestes de "yes" ou "no". Le système évaluera vos réponses et affichera un score basé sur la précision des réponses.
 
