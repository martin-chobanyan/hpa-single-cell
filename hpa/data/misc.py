def parse_string_label(label_str):
    return [int(i) for i in label_str.split('|')]


def get_single_label_subset(df):
    return df.loc[~df['Label'].str.contains('\|')]
