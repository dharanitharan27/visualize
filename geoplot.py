"""
GeoPlot Visualization Tool
---------------------------

This tool generates an interactive 3D CesiumJS-based visualization from a simulation's state trajectory.
It outputs:
- An HTML file to render the plot.
- A GeoJSON file representing the data over time.
"""

# Import necessary libraries
# re: Regular expressions for string operations
# json: For handling JSON data
# pandas: For timestamp generation and manipulation
# numpy: For numerical operations
# Template: For substituting values in the HTML template
# get_by_path: Helper function to extract nested properties from a dictionary

import re
import json
import pandas as pd
import numpy as np
from string import Template
from agent_torch.core.helpers import get_by_path

# HTML template for Cesium visualization
# This template defines the structure and behavior of the Cesium-based visualization.
# It includes functions for interpolating colors, determining pixel sizes, and processing time-series data.

geoplot_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Cesium Time-Series Heatmap Visualization</title>
    <script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
    <link href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css" rel="stylesheet" />
    <style>#cesiumContainer { width: 100%; height: 100%; }</style>
  </head>
  <body>
    <div id="cesiumContainer"></div>
    <script>
      Cesium.Ion.defaultAccessToken = '$accessToken';
      const viewer = new Cesium.Viewer('cesiumContainer');

      function interpolateColor(color1, color2, factor) {
        const result = new Cesium.Color();
        result.red = color1.red + factor * (color2.red - color1.red);
        result.green = color1.green + factor * (color2.green - color1.green);
        result.blue = color1.blue + factor * (color2.blue - color1.blue);
        result.alpha = '$visualType' == 'size' ? 0.2 : color1.alpha + factor * (color2.alpha - color1.alpha);
        return result;
      }

      function getColor(value, min, max) {
        const factor = (value - min) / (max - min);
        return interpolateColor(Cesium.Color.BLUE, Cesium.Color.RED, factor);
      }

      function getPixelSize(value, min, max) {
        const factor = (value - min) / (max - min);
        return 100 * (1 + factor);
      }

      function processTimeSeriesData(geoJsonData) {
        const timeSeriesMap = new Map();
        let minValue = Infinity;
        let maxValue = -Infinity;

        geoJsonData.features.forEach((feature) => {
          const id = feature.properties.id;
          const time = Cesium.JulianDate.fromIso8601(feature.properties.time);
          const value = feature.properties.value;
          const coordinates = feature.geometry.coordinates;

          if (!timeSeriesMap.has(id)) {
            timeSeriesMap.set(id, []);
          }
          timeSeriesMap.get(id).push({ time, value, coordinates });

          minValue = Math.min(minValue, value);
          maxValue = Math.max(maxValue, value);
        });

        return { timeSeriesMap, minValue, maxValue };
      }

      function createTimeSeriesEntities(timeSeriesData, startTime, stopTime) {
        const dataSource = new Cesium.CustomDataSource('AgentTorch Simulation');

        for (const [id, timeSeries] of timeSeriesData.timeSeriesMap) {
          const entity = new Cesium.Entity({
            id: id,
            availability: new Cesium.TimeIntervalCollection([
              new Cesium.TimeInterval({ start: startTime, stop: stopTime }),
            ]),
            position: new Cesium.SampledPositionProperty(),
            point: {
              pixelSize: '$visualType' == 'size' ? new Cesium.SampledProperty(Number) : 10,
              color: new Cesium.SampledProperty(Cesium.Color),
            },
            properties: {
              value: new Cesium.SampledProperty(Number),
            },
          });

          timeSeries.forEach(({ time, value, coordinates }) => {
            const position = Cesium.Cartesian3.fromDegrees(coordinates[0], coordinates[1]);
            entity.position.addSample(time, position);
            entity.properties.value.addSample(time, value);
            entity.point.color.addSample(time, getColor(value, timeSeriesData.minValue, timeSeriesData.maxValue));
            if ('$visualType' == 'size') {
              entity.point.pixelSize.addSample(time, getPixelSize(value, timeSeriesData.minValue, timeSeriesData.maxValue));
            }
          });

          dataSource.entities.add(entity);
        }

        return dataSource;
      }

      const geoJsons = $data;
      const start = Cesium.JulianDate.fromIso8601('$startTime');
      const stop = Cesium.JulianDate.fromIso8601('$stopTime');

      viewer.clock.startTime = start.clone();
      viewer.clock.stopTime = stop.clone();
      viewer.clock.currentTime = start.clone();
      viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
      viewer.clock.multiplier = 3600; // 1 hour per second

      viewer.timeline.zoomTo(start, stop);

      for (const geoJsonData of geoJsons) {
        const timeSeriesData = processTimeSeriesData(geoJsonData);
        const dataSource = createTimeSeriesEntities(timeSeriesData, start, stop);
        viewer.dataSources.add(dataSource);
        viewer.zoomTo(dataSource);
      }
    </script>
  </body>
</html>
"""

# Helper function to extract nested property from state based on path
# This function uses the get_by_path utility to navigate nested dictionaries.

def read_var(state, var):
    """Helper to extract nested property from state based on path."""
    return get_by_path(state, re.split("/", var))

# GeoPlot class
# This class encapsulates the logic for generating GeoJSON and HTML visualizations.
# It takes configuration and visualization options as input.

class GeoPlot:
    def __init__(self, config, options):
        """Initialize GeoPlot with config and visualization options."""
        self.config = config
        (
            self.cesium_token,
            self.step_time,
            self.entity_position,
            self.entity_property,
            self.visualization_type,
        ) = (
            options["cesium_token"],
            options["step_time"],
            options["coordinates"],
            options["feature"],
            options["visualization_type"],
        )

    # Render the trajectory to a GeoJSON and HTML visualization
    # This method processes the state trajectory to generate GeoJSON features and an HTML file.
    # It extracts coordinates and property values, generates timestamps, and constructs GeoJSON features.

    def render(self, state_trajectory):
        """Render the trajectory to a GeoJSON and HTML visualization."""
        coords, values = [], []
        name = self.config["simulation_metadata"]["name"]
        geodata_path, geoplot_path = f"{name}.geojson", f"{name}.html"

        # Extract coordinates and property values from final states
        for i in range(0, len(state_trajectory) - 1):
            final_state = state_trajectory[i][-1]
            coords = np.array(read_var(final_state, self.entity_position)).tolist()
            values.append(
                np.array(read_var(final_state, self.entity_property)).flatten().tolist()
            )

        start_time = pd.Timestamp.utcnow()

        # Generate timestamps spaced by step_time
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.step_time)
            for i in range(
                self.config["simulation_metadata"]["num_episodes"] *
                self.config["simulation_metadata"]["num_steps_per_episode"]
            )
        ]

        geojsons = []

        # Construct GeoJSON features for each coordinate
        for i, coord in enumerate(coords):
            features = []
            for time, value_list in zip(timestamps, values):
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [coord[1], coord[0]],
                    },
                    "properties": {
                        "value": value_list[i],
                        "time": time.isoformat(),
                    },
                })
            geojsons.append({"type": "FeatureCollection", "features": features})

        # Write GeoJSON file
        with open(geodata_path, "w", encoding="utf-8") as f:
            json.dump(geojsons, f, ensure_ascii=False, indent=2)

        # Fill the HTML template with real data and token
        # The HTML file is generated by substituting values into the Cesium template.
        # It includes the Cesium token, start and stop times, GeoJSON data, and visualization type.

        tmpl = Template(geoplot_template)
        with open(geoplot_path, "w", encoding="utf-8") as f:
            f.write(
                tmpl.substitute({
                    "accessToken": self.cesium_token,
                    "startTime": timestamps[0].isoformat(),
                    "stopTime": timestamps[-1].isoformat(),
                    "data": json.dumps(geojsons),
                    "visualType": self.visualization_type,
                })
            )
