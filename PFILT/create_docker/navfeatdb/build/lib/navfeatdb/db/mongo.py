

def add_rows_to_table(mongo_collection, fdf, desc):
    """
    Taking in a feature, which is a table.row of Class Feature, feed in \
    df and dg, which are 2 np.arrays returned from features_from_slice
    """
    rows = []
    for idx, row in fdf.iterrows():
        row = {
            "loc": {"type": "Point",
                    "coordinates": [row["lon"], row["lat"]]},
            'height': row['height'],
            'octave': row['octave'],
            'layer': row['layer'],
            'scale': row['scale'],
            'angle': row['angle'],
            'response': row['response'],
            'size': row['size'],
            'desc': desc[idx].tolist(),
        }
        rows.append(row)

    mongo_collection.insert_many(rows)
