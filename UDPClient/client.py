# !/usr/bin/python
# -*- coding: utf-8 -*-

# Client UDP conçu pour communiquer avec un serveur UDP

import socket

class UDPClient():


    def __init__(self, port=5035):
        self.port_ = port
        self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def listen(self):


        buf = 1024
        adresse = ('127.0.0.1', self.port_)
        # création du socket UDP
        while True:

            requete  = str({
                "legFrontLeftBot": 10.5, "legFrontLeftTop": 10.5, "shoulderFrontLeft": 10.5, "legFrontRightBot": 10.5,
                "legFrontRightTop": 10.5,"shoulderFrontRight": 10.5,"legBackLeftBot": 10.5,"legBackLeftTop": 10.5,
                "shoulderBackLeft": 10.5, "legBackRightBot": 10.5, "legBackRightTop": 10.5,"shoulderBackRight": 10.5
                })

            # test d'arrêt
            if requete == "":
                    break
            # envoi de la requête au serveur
            self.socket_.sendto(requete.encode(), adresse)

            # réception et affichage de la réponse
            #reponse, adr = self.socket_.recvfrom(buf)

            #print ("=> %s" % reponse)

            # fermeture de la connexion
            #self.socket_.close()
            print ("fin du client UDP")


if __name__ == "__main__":
    client = UDPClient()
    client.listen()