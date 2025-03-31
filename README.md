Class : 
    Team & Player : gére les teams du jeux qui contiennent deux players & gére les players
    Utils : contient quelque stu=ructure de donnée pour modéliser les couleurs et la valeur des cartes et les cartes elle même
    annonce_suit : gere la structure de donnée pour les annonce types carré tierce etc et une partie de leur gestion dans le jeu
    Main : class qui s'occupe de lancer une partie avec quatre joueur humain avec choix dans la console
        tout le court du jeux est dans cette class 
        phase d'annonce et phase de jeux (avec coinche, annonce des carré tierce etc)
        gére le lien entre le court du jeux et la classe game qui gére le jeu de coinche:
    Game:
        class centrale 
        contient toute les instances des autres classe pour faire fonctionner le jeu
        gere les fin de round de partie les carte legales a poser etc
    Coinche:
        contient plusieurs class:
            CoincheTable :
                gére la table de jeu
                contient les carte qui ont été poser et la levé en court 
                s'occupe du jeu des joueur, du decompte des points et de qui gagne
                reference qui pose les carte qui gagne les plis 
                contient l'annonce faiter pour la manche
            CoincheBid :
                correspond a une annonce contient quelle equipe part a quelle couleur pour quelle valeur etc
            CoincheDeck : 
                gére le jeux de carte : melange distribution et coupe
    modele_jeu : different modele de renforcement pour la phase de jeu lié a l'environement
    train-test : class pour lancer l'entrainement
