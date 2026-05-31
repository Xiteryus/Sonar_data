import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray, Empty
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import pandas as pd
import numpy as np
import math


class RadarSimulator(Node):
    def __init__(self):
        super().__init__('radar_simulator')
        # Donnees de SIMULATION : lignes jamais vues par le modele a l'entrainement.
        # -> les erreurs sont naturelles, pas besoin de bruit artificiel.
        self.sonar_data = pd.read_csv("sonar_simu.csv", header=None)

        self.env_pub = self.create_publisher(MarkerArray, '/environment', 10)
        self.beam_pub = self.create_publisher(Marker, '/sonar_beam_viz', 10)
        self.sonar_pub = self.create_publisher(Float32MultiArray, '/sonar/acoustic_signature', 10)
        # Signal envoye a ros_2d.py pour qu'il efface ses cercles de detection
        # a chaque nouvelle vague d'obstacles.
        self.clear_pub = self.create_publisher(Empty, '/sonar/clear_detections', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.theta = 0.0
        self.scan_speed = 1.5
        self.obstacles = []

        # --- NIVEAU DE BRUIT GLOBAL ---
        # Les donnees de simulation sont deja inconnues du modele,
        # donc des erreurs naturelles apparaissent meme SANS bruit (0.0).
        # Tu peux ajouter un petit bruit (ex: 0.02) pour rendre la tache encore plus dure.
        self.noise_std = 0.0

        # Compteur de generations : chaque vague d'obstacles recoit des IDs uniques
        # pour eviter que RViz reutilise/ecrase les marqueurs precedents.
        self.generation = 0

        self.create_environment()

        self.create_timer(0.05, self.update_simulation)
        self.create_timer(1.0, self.publish_environment)
        # --- REGENERATION : nouveaux obstacles toutes les 10 secondes ---
        self.create_timer(10.0, self.regenerate_environment)

    def create_environment(self):
        for i in range(10):
            angle = np.random.uniform(0, 2 * math.pi)
            radius = np.random.uniform(2.0, 6.0)

            is_mine = bool(np.random.choice([True, False]))
            target_label = 'M' if is_mine else 'R'

            subset = self.sonar_data[self.sonar_data[60] == target_label]
            row = subset.sample(1).iloc[0]

            # On stocke UNIQUEMENT les 60 features propres (float).
            # Le label et la distance sont stockés à part pour ne jamais être bruités.
            features = np.array(row[:60].values, dtype=float)

            self.obstacles.append({
                # ID unique a travers les generations : 1000*generation + i
                'id': self.generation * 1000 + i,
                'angle': angle,
                'radius': radius,
                'x': radius * math.cos(angle),
                'y': radius * math.sin(angle),
                'features': features,          # numpy array propre, 60 valeurs
                'true_label': 0.0 if is_mine else 1.0,  # 0 = Mine, 1 = Rocher
                'scanned_recently': False,
                'is_mine': is_mine
            })

    def regenerate_environment(self):
        # 1) Effacer TOUS les anciens cubes dans RViz avec un marqueur DELETEALL.
        clear_msg = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = "map"
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.ns = "true_environment"
        delete_all.action = Marker.DELETEALL
        clear_msg.markers.append(delete_all)
        self.env_pub.publish(clear_msg)

        # 1bis) Demander a ros_2d.py d'effacer ses cercles de detection.
        self.clear_pub.publish(Empty())

        # 2) Generer une nouvelle vague d'obstacles.
        self.generation += 1
        self.obstacles = []
        self.create_environment()

        self.get_logger().info(
            f"=== Nouvelle vague d'obstacles (generation {self.generation}) ===")

        # 3) Publier immediatement le nouvel environnement.
        self.publish_environment()

    def publish_environment(self):
        msg = MarkerArray()
        for obs in self.obstacles:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "true_environment"
            marker.id = obs['id']
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obs['x']
            marker.pose.position.y = obs['y']
            marker.pose.position.z = 0.0
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4

            marker.color.a = 0.8
            if obs['is_mine']:
                marker.color.r = 1.0
                marker.color.g = 0.2
                marker.color.b = 0.2  # Rouge
            else:
                marker.color.r = 0.2
                marker.color.g = 0.2
                marker.color.b = 1.0  # Bleu

            msg.markers.append(marker)
        self.env_pub.publish(msg)

    def update_simulation(self):
        dt = 0.05
        self.theta = (self.theta + self.scan_speed * dt) % (2 * math.pi)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'sonar_beam_link'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(self.theta / 2.0)
        t.transform.rotation.w = math.cos(self.theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

        beam = Marker()
        beam.header.frame_id = "sonar_beam_link"
        beam.header.stamp = self.get_clock().now().to_msg()
        beam.ns = "sonar_beam"
        beam.id = 100
        beam.type = Marker.ARROW
        beam.action = Marker.ADD
        beam.scale.x = 7.0
        beam.scale.y = 0.05
        beam.scale.z = 0.05
        beam.color.a = 0.4
        beam.color.r = 0.0
        beam.color.g = 1.0
        beam.color.b = 1.0
        self.beam_pub.publish(beam)

        for obs in self.obstacles:
            angle_diff = (obs['angle'] - self.theta + math.pi) % (2 * math.pi) - math.pi

            if abs(angle_diff) < 0.1:
                if not obs['scanned_recently']:
                    obs['scanned_recently'] = True

                    # --- BRUIT GAUSSIEN UNIFORME SUR LES 60 BANDES ---
                    # Appliqué de façon IDENTIQUE aux Mines et aux Rochers.
                    # On repart toujours du signal propre -> pas d'accumulation.
                    noise = np.random.normal(0.0, self.noise_std, size=60)
                    noisy_features = obs['features'] + noise

                    # Les valeurs du dataset sont bornées dans [0, 1] : on respecte ça.
                    noisy_features = np.clip(noisy_features, 0.0, 1.0)

                    # On construit le message : 60 features + label + distance
                    data = noisy_features.tolist()
                    data.append(obs['true_label'])   # index 60
                    data.append(obs['radius'])        # index 61

                    msg = Float32MultiArray()
                    msg.data = [float(v) for v in data]
                    self.sonar_pub.publish(msg)
            else:
                obs['scanned_recently'] = False


def main(args=None):
    rclpy.init(args=args)
    node = RadarSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()