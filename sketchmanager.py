from scipy.spatial import distance
from datasketches import kll_floats_sketch
from datetime import datetime

class SketchManager:
    def __init__(self):
        self.model_stats = {}
        self.image_stats = {}
        self.samples = {}
        self.image_stats_ref = {}

        self.sensor_image_stats = {}
        self.sensor_model_stats = {}
        self.sensor_sample_stats = {}


        self.sensor_image_images = {}
        self.sensor_model_images = {}
        self.sensor_sample_images = {} 

    def read_sketch(self, file_path):
        with open(file_path, 'rb') as f:
            sketch = kll_floats_sketch.deserialize(f.read())
        return sketch

    def _add_to_stats(self, stats, last_updated, metric, sub_metric, sketch):
        if last_updated not in stats:
            stats[last_updated] = {}
        if metric not in stats[last_updated] and sub_metric:
            stats[last_updated][metric] = {}
            stats[last_updated][metric][sub_metric] = kll_floats_sketch()
        elif metric in stats[last_updated] and sub_metric:
            if sub_metric not in stats[last_updated][metric]:
                stats[last_updated][metric][sub_metric] = kll_floats_sketch()
            else:
                stats[last_updated][metric][sub_metric].merge(sketch)
        
        if metric not in stats[last_updated] and not sub_metric:
            stats[last_updated][metric] = kll_floats_sketch()
        elif metric in stats[last_updated] and not sub_metric:    
            stats[last_updated][metric].merge(sketch)

    def add_sketch(self, sensor_id, timestamp, file_updated_timestamp, file_type, metric, sub_metric, file_path):
        sketch = None
        if file_path.endswith('bin'):
            sketch = self.read_sketch(file_path)
        if file_type == "modelstats":
            if sketch:
                self._add_to_stats(self.model_stats, timestamp, metric, sub_metric, sketch)
                self.add_sensors(self.sensor_model_stats, sensor_id, file_updated_timestamp, metric, sub_metric, file_path)
            else:
                self.add_sensors(self.sensor_model_images, sensor_id, file_updated_timestamp, metric, sub_metric, file_path)
        elif file_type == "imagestats":
            if sketch:
                self._add_to_stats(self.image_stats, timestamp, metric, sub_metric, sketch)
                self.add_sensors(self.sensor_image_stats, sensor_id, file_updated_timestamp, metric, sub_metric, file_path)
            else:
                self.add_sensors(self.sensor_image_images, sensor_id, file_updated_timestamp, metric, sub_metric, file_path)

        elif file_type == "samples":
            if sketch:
                self._add_to_stats(self.samples, timestamp, metric, sub_metric, sketch)
                self.add_sensors(self.sensor_sample_stats, sensor_id, file_updated_timestamp, metric, sub_metric, file_path)
            else:
                self.add_sensors(self.sensor_sample_images, sensor_id, file_updated_timestamp, metric, sub_metric, file_path)
    
    def add_sensors(self, stats, sensor_id, timestamp, metric, sub_metric, file_path):
        if file_path.endswith('.bin'):
            file_object = self.read_sketch(file_path)
            if sensor_id not in stats:
                stats[sensor_id] = {}
            if timestamp not in stats[sensor_id].keys():
                stats[sensor_id][timestamp] = {}
            if metric not in stats[sensor_id][timestamp] and sub_metric:
                stats[sensor_id][timestamp][metric] = {}
                stats[sensor_id][timestamp][metric][sub_metric] = file_object
            elif metric in stats[sensor_id][timestamp] and sub_metric:
                if sub_metric not in stats[sensor_id][timestamp][metric]:
                    stats[sensor_id][timestamp][metric][sub_metric] = file_object
        
            if metric not in stats[sensor_id][timestamp] and not sub_metric:
                stats[sensor_id][timestamp][metric] = file_object

        elif file_path.endswith('.png'):
            file_object = file_path
            if sensor_id not in stats:
                stats[sensor_id] = {}
            if timestamp not in stats[sensor_id].keys():
                stats[sensor_id][timestamp] = {}
            if metric not in stats[sensor_id][timestamp] and sub_metric:
                stats[sensor_id][timestamp][metric] = {}
                if sub_metric not in stats[sensor_id][timestamp][metric]:
                    stats[sensor_id][timestamp][metric][sub_metric]=[]
                stats[sensor_id][timestamp][metric][sub_metric].append(file_object)
            elif metric in stats[sensor_id][timestamp] and sub_metric:
                if sub_metric not in stats[sensor_id][timestamp][metric]:
                    stats[sensor_id][timestamp][metric][sub_metric]=[]
                    stats[sensor_id][timestamp][metric][sub_metric].append(file_object)
                else:
                    stats[sensor_id][timestamp][metric][sub_metric].append(file_object)
            elif metric not in stats[sensor_id][timestamp] and not sub_metric:
                stats[sensor_id][timestamp][metric] = {}
                if isinstance(stats[sensor_id][timestamp][metric], list):
                    stats[sensor_id][timestamp][metric].append(file_object)
                else:
                    stats[sensor_id][timestamp][metric] = []
                    stats[sensor_id][timestamp][metric].append(file_object)
            elif metric in stats[sensor_id][timestamp] and not sub_metric:
                if isinstance(stats[sensor_id][timestamp][metric], list):
                    stats[sensor_id][timestamp][metric].append(file_object)
                else:
                    stats[sensor_id][timestamp][metric] = []
                    stats[sensor_id][timestamp][metric].append(file_object)


    def add_reference_sketch(self, last_updated, file_type, metric, sub_metric, file_path):
        sketch = self.read_sketch(file_path)
        if last_updated not in self.image_stats_ref:
            self.image_stats_ref[last_updated] = {}
        if metric not in self.image_stats_ref[last_updated] and sub_metric:
            self.image_stats_ref[last_updated][metric] = {}
            self.image_stats_ref[last_updated][metric][sub_metric] = sketch
        elif metric in self.image_stats_ref[last_updated] and sub_metric:    
            if sub_metric not in self.image_stats_ref[last_updated][metric]:
                self.image_stats_ref[last_updated][metric][sub_metric]= sketch
        elif metric not in self.image_stats_ref and not sub_metric:
            self.image_stats_ref[last_updated][metric] = sketch

    def display_distance_metrics(stats1, stats2):
        distance_metrics = {}
        for timestamp, stats in stats1.items():
            distance_metrics[timestamp] = {}
            for metric, sub_metrics in stats.items():
                if metric in stats2[stats2.keys()[0]]:
                    distance_metrics[metric] = {}
                    if sub_metrics:  # Check if there are sub-metrics
                        for sub_metric, sketch1 in sub_metrics.items():
                            if sub_metric in stats2[metric]:
                                # Check if the sub-metric exists in stats2
                                sketch2 = stats2[metric][sub_metric]
                                # Compute distance metric between sketches
                                dist = distance.jensenshannon(sketch1.get_pmf(), sketch2.get_pmf())
                                distance_metrics[metric][sub_metric] = dist
                    elif not sub_metrics and metric in stats2 and not stats2[metric]:
                        # If no sub-metrics in both stats1 and stats2
                        sketch1 = stats1[metric]
                        sketch2 = stats2[metric]
                        dist = distance.jensenshannon(sketch1.get_pmf(), sketch2.get_pmf())
                        distance_metrics[metric] = dist
        st.subheader("Distance Metrics")
        st.write(distance_metrics)

    def compute_distance_metrics(self, file_type1, file_type2):
        if file_type1 == 'modelstats' and file_type2 == 'modelstats':
            return self._compute_distance_metric(self.model_stats, self.model_stats)
        elif file_type1 == 'imagestats' and file_type2 == 'imagestats':
            return self._compute_distance_metric(self.image_stats, self.image_stats)
        else:
            raise ValueError("Invalid file types for computing distance metrics")

# Example usage
# sketch_manager = SketchManager()
# sketch_manager.add_sketch(sensor_id, 'modelstats', 'accuracy', None, 'path_to_model_accuracy_sketch')
# sketch_manager.add_sketch(sensor_id, 'imagestats', 'image_quality', 'brightness', 'path_to_image_brightness_sketch')
# distance_metrics = sketch_manager.compute_distance_metrics('modelstats', 'imagestats')
# print(distance_metrics)

