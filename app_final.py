# Standard imports
import os
import io

import pandas as pd
import numpy as np
from PIL import Image

# Geospacial processing packages
import geopandas as gpd

import shapely
import rasterio as rio
from rasterio.plot import show
import rasterio.mask
from shapely.geometry import box

# Mapping and plotting libraries
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as cl

# Model
import torch
from torchvision import datasets, models, transforms

from math import pi

from bokeh.palettes import Category20c, Category20
from bokeh.plotting import figure
from bokeh.transform import cumsum

import panel as pn
pn.extension()


# Getting the files

tiles_dic = {"Liverpool":gpd.read_file("Liverpool20250511.zip"),
             "Norfork": gpd.read_file("Norfolk20250512.zip"),
             "NELincolnshire": gpd.read_file("NELincolnshire20250510.zip"),
             "Cornwall": gpd.read_file("Cornwall20250518.zip"),
             "Worcestershire": gpd.read_file("Worcestershire20250619.zip"),
             }


# LULC Classes
classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River"
]



from typing import List
# Instantiate map centered on the centroid
def folium_Map(tiles, classes: List = classes):
  map = folium.Map(zoom_start=10,world_copy_jump=True)

  colors = {
  'AnnualCrop' : 'lightgreen',
  'Forest' : 'forestgreen',
  'HerbaceousVegetation' : 'yellowgreen',
  'Highway' : 'black',
  'Industrial' : 'red',
  'Pasture' : 'mediumseagreen',
  'PermanentCrop' : 'chartreuse',
  'Residential' : 'magenta',
  'River' : 'dodgerblue'
}

  classes_plot = {}
  for classe in classes:
    classes_plot[classe] = colors[classe]

  # Add Google Satellite basemap
  folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
  ).add_to(map)

  # Add LULC Map with legend
  legend_txt = '<span style="color: {col};">{txt}</span>'
  for label, color in classes_plot.items():

    # Specify the legend color
    name = legend_txt.format(txt=label, col=color)
    feat_group = folium.FeatureGroup(name=name)

    # Add GeoJSON to feature group
    subtiles = tiles[tiles.pred==label]
    if len(subtiles) > 0:
      folium.GeoJson(
          subtiles,
          style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            'color': 'black',
            'weight': 0,
            'fillOpacity': 0.5,
          },
          name='LULC Map'
      ).add_to(feat_group)
      map.add_child(feat_group)

  folium.LayerControl().add_to(map)
  map.fit_bounds(map.get_bounds())
  return map

def update_map(city, lista, width=700, height=700):
    # Filter the DataFrame for the selected country
    folium_map = folium_Map(tiles_dic[city], lista)
    # Panel doesn't directly render Folium maps, so we need to render it as HTML
    return pn.pane.HTML(folium_map._repr_html_(), width=width, height=height)

# Park Types
lulc_list = sorted(classes)
lulc_cbg = pn.widgets.CheckBoxGroup(name="Filter Landcover type", value=lulc_list, options=lulc_list)

# Cities Selector
cities_ls = sorted(["Liverpool","Norfork","NELincolnshire","Cornwall","Worcestershire"])
cities_sl = pn.widgets.Select(name="Select City", options=cities_ls)

#Filsters
filters = pn.Column(
    cities_sl,
    lulc_cbg
)

main_map = pn.bind(update_map,cities_sl, lulc_cbg, 800, 800)



#Pie chart__

# Pie chat to show the selected administrative boundary number of parks
def update_chart(city):
    #filter
    tile = tiles_dic[city]
    data_plot = tile["pred"].value_counts(normalize=True) * 100
    data_plot = data_plot.reset_index(name='Percentage').rename(columns={'pred':'Landcover'})
    data_plot["angle"] = data_plot["Percentage"]/data_plot["Percentage"].sum()*2*pi
    # Assigning colors
    if len(data_plot) in Category20c:
        colors = Category20c[len(data_plot)]
    else:
        colors = Category20c[3][:len(data_plot)]  # Pick 3 colors and slice

    data_plot["color"] = colors

    data_plot['legend_label'] = data_plot['Landcover'] + ": " + round(data_plot['Percentage'],2).astype(str)+"%"

    p = figure(height=300, title = "Percenge occupied by", toolbar_location=None, width=550,
               tools="hover", tooltips="@Landcover: @Percentage", x_range=(-0.3,1.0))

    r = p.wedge(x=0, y=1, radius=0.25,
                start_angle=cumsum("angle",include_zero=True), end_angle=cumsum("angle"),
                line_color="white", fill_color="color", legend_field="legend_label", source=data_plot)

    p.axis.visible=False
    p.grid.grid_line_color = None
    p.legend.location = "right"#"top_right"
    #p.add_layout(p.legend[0], 'right')
    bokeh_pane = pn.pane.Bokeh(p, theme="dark minimal")

    return bokeh_pane


pie_chart = pn.bind(update_chart,
                   cities_sl
                   )


# Creatin the layout

# Using a box that confine the content
FILTERS = pn.WidgetBox(
    '# Filters:',
    cities_sl,
    'Filter Landcover type:',
    lulc_cbg,
    styles={'overflow': 'auto', 'border': '1px solid lightgray'} # Uses scrolling when there is many filters
    #sizing_mode='stretch_both',  # or 'stretch_width' or 'fixed'
    #max_width=250                # adjust as needed
)


MAIN_MAP = pn.WidgetBox( pn.Column(main_map,#height=400,width=600, sizing_mode=None,
                        styles={'overflow': 'auto', 'border': '1px solid lightgray'}))

PAI_STATS = pn.WidgetBox(pie_chart)


tab1 = pn.GridSpec(width=1100, height=700, nrows=3, ncols=3)

tab1[0:3, 0] = FILTERS  # Filters
tab1[0:2, 1:3] =  MAIN_MAP # Main mapt
tab1[2, 1:3] =  PAI_STATS # pie chart



# Tab 2 - Classify uplaod images:


device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_tiles(image_file, size=64):
    """Genarates size*size polygon tiles.
    """

    # Open the raser image using rasterio
    raster = rio.open(image_file)
    width, height = raster.shape

    # Create a dictionary which will conatin our 64Z64 px polygon tiles
    # Later convert this dict into GeoPandas DataFrame
    geo_dict = {"id":[],"geometry":[]}
    index = 0

    # Do a sliding windows across the raste image
    for w in range(0, width, size):
      for h in range(0, height, size):
          # Create a Window of your disired size
          window = rio.windows.Window(h, w, size, size)

          # Get the georeferenced windowss bounds
          bbox = rio.windows.bounds(window, raster.transform)

          # Create a shapely geometry from the bounding box
          bbox = box(*bbox)

          # Create a unique id for each geometry
          uid = str(index)

          # Update dictionary
          geo_dict["id"].append(uid)
          geo_dict["geometry"].append(bbox)

          index += 1

    # Cast dictionary as a GeoPandas DataFrame
    results = gpd.GeoDataFrame(pd.DataFrame(geo_dict))

    # Set CRS to EPSG: 4326
    results.set_crs("EPSG:4326", inplace=True)

    raster.close()

    return results



transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_crop(image, shape, classes, model, show=False):
    """
    Generate prediction from a raster image using a geometry crop,
    """
    with rio.open(image) as img:
        # Crop source image using polygon shape
        out_image, out_transform = rio.mask.mask(img, shape, crop=True)

        # Crop out black (zero) border
        _, x_nonzero, y_nonzero = np.nonzero(out_image)
        out_image = out_image[
            :,
            np.min(x_nonzero):np.max(x_nonzero),
            np.min(y_nonzero):np.max(y_nonzero)
        ]

        # Convert to PIL image for model input
        np_img = np.moveaxis(out_image, 0, -1)  # (bands, H, W) -> (H, W, bands)

        pil_img = Image.fromarray(np_img.astype(np.uint8))

        # Apply transforms and make prediction
        input_tensor = transform(pil_img).to(device)
        output = model(input_tensor.unsqueeze(0))
        _, pred = torch.max(output, 1)
        label = str(classes[int(pred[0])])

        if show:
            pil_img.show(title=label)

        return label



file_input = pn.widgets.FileInput(accept='.tif')

message = pn.widgets.StaticText(name='Message', value='Nothing')

# Get file when imputed
def update_map1(input_file=None):
  if input_file:
    message.value = "Uploaded"
    # Convert the uploaded binary data to a file-like object
    file_bytes = io.BytesIO(file_input.value)
    # Open with rasterio
    image = rio.open(file_bytes)

    message.value = "Creating tiles"
    tiles = generate_tiles(file_bytes, size=64)

    # LULC Classes
    classes = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River"
    ]

    message.value = "Loading model"
    #Get the model
    path_drive = "RESNET50_0001_SDG-RGB_B32_WW_EP150_S224_9CL.pth"

    model_2 = models.resnet50()
    model_2.fc = torch.nn.Linear(in_features=model_2.fc.in_features, out_features=len(classes))
    model_2.load_state_dict(torch.load(f=path_drive, map_location=device))
    model_2 = model_2.to(device)

    model_2.eval()



    # Get label

    #
    message.value = "Making prediction: "
    labels = [] # Stor prediction
    #for index in tqdm(range(len(tiles)), total=len(tiles)):
    l = len(tiles)
    for index in range(l):
      message.value = f"Making prediction: {index}/{l}"
      try:
        # Use model_2 instead of model
        label = predict_crop(file_bytes, [tiles.iloc[index]['geometry']], classes, model_2)
        labels.append(label)
      except:
        labels.append("NoData")
        continue

    # Create tile
    tiles['pred'] = labels


    message.value = "Creating map..."

    tiles = tiles[tiles['pred'] != 'NoData']
    # We map each class to a corresponding color
    colors = {
      'AnnualCrop' : 'lightgreen',
      'Forest' : 'forestgreen',
      'HerbaceousVegetation' : 'yellowgreen',
      'Highway' : 'black',
      'Industrial' : 'red',
      'Pasture' : 'mediumseagreen',
      'PermanentCrop' : 'chartreuse',
      'Residential' : 'magenta',
      'River' : 'dodgerblue',
      'SeaLake' : 'blue'
    }
    tiles['color'] = tiles["pred"].apply(
      lambda x: cl.to_hex(colors.get(x))
    )
    # Divide Tiles





    # Instantiate map centered on the centroid
    map = folium.Map(zoom_start=10)


    # Add Google Satellite basemap
    folium.TileLayer(
          tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
          attr = 'Google',
          name = 'Google Satellite',
          overlay = True,
          control = True
    ).add_to(map)

    # Add LULC Map with legend
    legend_txt = '<span style="color: {col};">{txt}</span>'
    for label, color in colors.items():

      # Specify the legend color
      name = legend_txt.format(txt=label, col=color)
      feat_group = folium.FeatureGroup(name=name)

      # Add GeoJSON to feature group
      subtiles = tiles[tiles.pred==label]
      if len(subtiles) > 0:
        folium.GeoJson(
            subtiles,
            style_function=lambda feature: {
              'fillColor': feature['properties']['color'],
              'color': 'black',
              'weight': 0,
              'fillOpacity': 0.5,
            },
            name='LULC Map'
        ).add_to(feat_group)
        map.add_child(feat_group)

    folium.LayerControl().add_to(map)
    # Reset zoom with data available
    map.fit_bounds(map.get_bounds())
    message.value = "Done!"
    return pn.pane.HTML(map._repr_html_(), width=900, height=900)
    # Classify
    # Process

    # Return map
"""
"""
# Attach callback
#file_input.param.watch(update_plot, 'value')
mapa= pn.bind(update_map1, file_input)
# Layout
main_area1=pn.Column(
    "# Upload GeoTIFF and View",
    file_input,
    mapa,
    message
)



MAIN=pn.WidgetBox(main_area1)


tab2 = pn.GridSpec(width=1100, height=700, nrows=3, ncols=3)

tab2[0:3, 0:3] = MAIN 


#title = "COMP702 - Landcover Decision Support Tool"



#title = "Panel Demo - Image Classification"
#pn.template.BootstrapTemplate(
#    title=title,
#    main=main,
#    main_max_width="min(50%, 698px)",
#    header_background="#F08080",
#).servable(title=title)


dashboard = pn.Tabs(("Monitor", tab1),("I have my on file", tab2))




# Use a template
appf = pn.template.BootstrapTemplate(
    title="COMP702 - Landcover Decision Support Tool",
    sidebar=[],          #
    main=[dashboard]  #
)

appf.servable()