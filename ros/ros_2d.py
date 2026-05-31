import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty
from visualization_msgs.msg import Marker
import joblib
import numpy as np
import csv
import os


class SonarClassifier2DNode(Node):
    def __init__(self):
        super().__init__('sonar_classifier_2d_node')
        self.get_logger().info("IA Sonar prête. En attente du balayage...")
        self.model = joblib.load('sonar_svm_model.pkl')
        self.scaler = joblib.load('sonar_scaler.pkl')

        self.subscription = self.create_subscription(
            Float32MultiArray, '/sonar/acoustic_signature', self.listener_callback, 10)
        self.marker_pub = self.create_publisher(Marker, '/sonar/detections_2d_viz', 10)
        # Ecoute le signal de simu_env.py pour effacer les cercles a chaque nouvelle vague.
        self.clear_sub = self.create_subscription(
            Empty, '/sonar/clear_detections', self.clear_callback, 10)
        self.marker_id = 0

        # --- INITIALISATION DU FICHIER CSV ---
        self.csv_filename = 'simulation_results.csv'
        file_exists = os.path.isfile(self.csv_filename)

        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                header = [f'freq_{i}' for i in range(60)] + \
                         ['vraie_nature', 'prediction_ia', 'est_correct', 'distance']
                writer.writerow(header)
                self.get_logger().info(f"Nouveau fichier d'enregistrement créé : {self.csv_filename}")
            else:
                self.get_logger().info(f"Enregistrement dans le fichier existant : {self.csv_filename}")

    def clear_callback(self, msg):
        # Efface tous les cercles de detection dans RViz (nouvelle vague d'obstacles).
        marker = Marker()
        marker.header.frame_id = "sonar_beam_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ia_classification"
        marker.action = Marker.DELETEALL
        self.marker_pub.publish(marker)

        # Remise a zero du compteur d'id pour la nouvelle vague.
        self.marker_id = 0
        self.get_logger().info("--- Cercles de detection effaces (nouvelle vague) ---")

    def listener_callback(self, msg):
        data = np.array(msg.data, dtype=float)

        # Le simulateur envoie 60 features + label + distance = 62 valeurs
        if len(data) != 62:
            self.get_logger().warn(f"Message ignoré : {len(data)} valeurs reçues (62 attendues).")
            return

        # Extraction stricte
        features = data[:60].reshape(1, -1)
        true_label = int(round(data[60]))  # 0 = Mine, 1 = Rocher
        distance = float(data[61])

        # Prédiction (mêmes 60 bandes que celles vues à l'entraînement)
        features_scaled = self.scaler.transform(features)
        prediction = int(self.model.predict(features_scaled)[0])

        is_mine_pred = (prediction == 0)
        is_correct = (prediction == true_label)

        # Logique d'évaluation textuelle
        if is_correct:
            nature = "MINE" if is_mine_pred else "ROCHER"
            self.get_logger().info(f"[SUCCES] L'IA a bien identifie un(e) {nature}.")
        else:
            fausse_nature = "MINE" if is_mine_pred else "ROCHER"
            vraie_nature = "ROCHER" if is_mine_pred else "MINE"
            self.get_logger().error(
                f"[ERREUR] L'IA a confondu un(e) {vraie_nature} avec un(e) {fausse_nature} !")

        # --- ENREGISTREMENT DANS LE CSV ---
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            row_data = features[0].tolist() + \
                [true_label, prediction, int(is_correct), round(distance, 2)]
            writer.writerow(row_data)

        # Affichage RViz
        self.publish_rviz_2d_marker(is_mine_pred, is_correct, distance)

    def publish_rviz_2d_marker(self, is_mine_pred, is_correct, distance):
        marker = Marker()
        marker.header.frame_id = "sonar_beam_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ia_classification"
        marker.id = self.marker_id
        self.marker_id += 1

        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = float(distance)
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.05
        marker.color.a = 1.0

        if not is_correct:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0  # Jaune = erreur
        elif is_mine_pred:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0  # Rouge = mine
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0  # Bleu = rocher

        marker.lifetime = rclpy.duration.Duration(seconds=4.0).to_msg()
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = SonarClassifier2DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()