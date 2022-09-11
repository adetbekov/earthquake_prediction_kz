import json
import dvc.api
import argparse
import datetime
import geopandas
import numpy as np
import pandas as pd


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

    
def fillna(df, strategy):
    if strategy == "mean":
        return df.fillna(df.mean())
    elif strategy == "min":
        return df.fillna(df.min())
    
    return df.fillna(0)


def collect(path, na_strategy, mag_max_target_th, roll_years_agg):
    with open(path) as f:
        data = json.load(f)
    
    metadata = {
        "initial": data["metadata"]
    }
    
    ## Reading data
    df = pd.json_normalize(data["features"])
    df = df.drop(["type"], axis=1)
    
    ## Filter earthquakes only
    type_counts = df["properties.type"].value_counts()
    metadata.update({"type_counts": type_counts.to_dict()})
    df = df[df["properties.type"] == "earthquake"]
    
    ## Normalize time
    df["time"] = df["properties.time"].apply(lambda x: datetime.datetime.fromtimestamp(round(int(x)*0.001)))
    df["updated"] = df["properties.updated"].apply(lambda x: datetime.datetime.fromtimestamp(round(int(x)*0.001)))
    df["year"] = df["time"].apply(lambda x: int(str(x)[:4]))
    
    ## Distribution by years
    
    
    ## Converting to geopandas df
    df["longitude"] = df["geometry.coordinates"].apply(lambda x: x[0])
    df["latitude"] = df["geometry.coordinates"].apply(lambda x: x[1])

    gdf = geopandas.GeoDataFrame(
        df,
        geometry = geopandas.points_from_xy(df["longitude"], df["latitude"])
    )

#     world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#     ax = world[world.iso_a3 == 'KAZ'].plot(color='white', edgecolor='black')
    
    ## Save to enrich coordinates with administrative areas names
#     df.to_csv("artifacts/preprocess/enrich_administrative_areas.csv", sep="\t", index=False)
    
    ## Read file with regions
    geocoded_earthquakes = pd.read_csv("data/geocoded_areas.csv", sep="\t")
    area_counts = geocoded_earthquakes.value_counts("REGION")
    metadata.update({"area_counts": area_counts.to_dict()})
    
    ## Join with regions
    geocoded_earthquakes["latitude"] = geocoded_earthquakes["lat"]
    geocoded_earthquakes["longitude"] = geocoded_earthquakes["lng"]

    geocoded_earthquakes = geocoded_earthquakes[["latitude", "longitude", "REGION"]].drop_duplicates()

    gdf_with_region = geopandas.GeoDataFrame(
        geocoded_earthquakes, 
        geometry = geopandas.points_from_xy(
            geocoded_earthquakes["longitude"], 
            geocoded_earthquakes["latitude"]
        )
    )
    
    data = geopandas.sjoin_nearest(gdf, gdf_with_region.drop(["latitude", "longitude"], axis=1))
    
    region_na_counts = data["REGION"].isna().value_counts()
    metadata.update({"region_na_counts": region_na_counts.to_dict()})
    
    data = data[~data["REGION"].isna()]
    
    ## Normalize main features
    data["depth"] = data["geometry.coordinates"].apply(lambda x: x[2])
    data["mag"] = data["properties.mag"]
    
    ## Aggregate features
    agg_funcs = ['mean', 'count', "min", "median", "max", "std"]
    agg = data[[
        "mag", "year", "REGION", "depth"
    ]].groupby(["REGION", "year"]).agg({
        "mag": agg_funcs,
        "depth": agg_funcs,
    })
    
    train = agg.reset_index()
    train.columns = ['_'.join(col) for col in train.columns]
    
    train = train.sort_values(by=["REGION_", "year_"])
    
    year_boundary = {
        "min_year": train['year_'].min(),
        "max_year": train['year_'].max()
    }
    metadata.update({"year_boundary": year_boundary})
    year_range = np.arange(year_boundary["min_year"], year_boundary["max_year"]+1)
    
    ## Cartesian product
    matrix = np.array(np.meshgrid(train["REGION_"].unique(), year_range)).T.reshape(-1,2)
    matrix_pd = pd.DataFrame(matrix, columns = ["REGION_", "year_"])
    train_cj = matrix_pd.merge(train, on=["REGION_", "year_"], how='outer')
    
    ## fill na
    train_cj = fillna(train_cj, na_strategy)
    
    ## Target
    train_cj["TARGET"] = train_cj["mag_max"] > mag_max_target_th
    
    ## Agg windows
    for i in range(1, 6):
        for n in ["mag", "depth"]:
            for m in agg_funcs:
                train_cj['{}_{}_ylag'.format(n, m, i)] = train_cj.groupby(["REGION_"])[n+"_"+m].apply(lambda v: v.shift(i))

    for n in ["mag", "depth"]:
        for m in agg_funcs:
            for LAG in roll_years_agg:
                train_cj['rollsum_{}y_{}_{}'.format(LAG, n, m)] = train_cj.groupby(["REGION_"])[n+"_"+m].apply(lambda v: v.shift(1).rolling(LAG).sum())
                train_cj['rollmin_{}y_{}_{}'.format(LAG, n, m)] = train_cj.groupby(["REGION_"])[n+"_"+m].apply(lambda v: v.shift(1).rolling(LAG).min())
                train_cj['rollmax_{}y_{}_{}'.format(LAG, n, m)] = train_cj.groupby(["REGION_"])[n+"_"+m].apply(lambda v: v.shift(1).rolling(LAG).max())
                train_cj['rollmean_{}y_{}_{}'.format(LAG, n, m)] = train_cj.groupby(["REGION_"])[n+"_"+m].apply(lambda v: v.shift(1).rolling(LAG).mean())
                
    target_size_counts = train_cj["TARGET"].value_counts(1)
    metadata.update({"target_size": target_size_counts.to_dict()[True]})
    
    zero_mag_counts = (train_cj["mag_max"] == 0).value_counts(1)
    metadata.update({"zero_mag": zero_mag_counts.to_dict()[True]})
    
    ## Drop leaks
    drop_leaks = [
        f"{feature}_{agg}" for agg in agg_funcs for feature in ["mag", "depth"] if (f"{feature}_{agg}" != "mag_max")
    ]
    train_cj = train_cj.drop(drop_leaks, axis=1)
    metadata.update({
        "final": {
            "shape": train_cj.shape
        }
    })
    
    return train_cj, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('na_strategy', type=str, help='With what to fill na values [mean, min, zero]')
    args = parser.parse_args()
    na_strategy = args.na_strategy
    
    params = dvc.api.params_show()

    data, metadata = collect(
        path = "data/data.json",
        na_strategy = na_strategy,
        mag_max_target_th = params["target"]["mag_max_target_th"],
        roll_years_agg = params["preprocess"]["roll_years_agg"]
    )
    
    data.to_csv(f"artifacts/{na_strategy}/data.csv", sep=";", index=False)
    with open(f"artifacts/{na_strategy}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        