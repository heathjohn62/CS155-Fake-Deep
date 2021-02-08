# Input: pandas dataframe containing data
# Output: pandas dataframe containing encoded data
# USAGE: add lines "in_data["NAME"] = ... " to encode additional columns
def label_encoding(in_data):
    # label_encoder object knows how to understand word labels.
    label_encoder = LabelEncoder()
    # Encode labels in column 'DATE'.
    in_data['DATE']= label_encoder.fit_transform(in_data['DATE'])
    # Encode labels in column 'SOURCE_REPORTING_UNIT_NAME'.
    in_data['SOURCE_REPORTING_UNIT_NAME']= label_encoder.fit_transform(in_data['SOURCE_REPORTING_UNIT_NAME'])

    return in_data
