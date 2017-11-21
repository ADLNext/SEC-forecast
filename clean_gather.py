import pandas as pd

data_dir = 'data/analysis/'

file_names = [
    '411_1',
    '411_2',
    '412',
    '413',
    '414',
    '415',
    '416',
    '421',
    '422',
    '423',
    '424',
    '425',
    '431',
    '432',
    '433',
    '434',
    '441_1',
    '441_2',
    '442_1',
    '442_2',
    '443_1',
    '443_2',
    '444'
]

df = pd.concat([
    pd.read_excel(data_dir+name+ '.xlsx', sheetname=name) for name in file_names
], axis = 0)

print('Files read, shape before cleaning:', df.shape)
print('Preparing material dictionary to replace DELETED items')

material_dict = df.set_index('Rationalized Material Num')['Mat.Desc.'].to_dict()

df = df [[
    'Mat.Desc.',
    'Quantity.1',
    'Department',
    'Project type',
    'WBS Element',
    'Work order number',
    'Amount.1',
    'Document Date'
]]

df.columns = ['Material', 'Quantity', 'Department', 'Project number', 'WBS entry', 'Order number', 'Value', 'Date']

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

print('Date transformation complete')

df_meta = pd.read_excel('data/Metadata.xlsx', sheetname='Project description')
df_meta['Project category'].dropna(inplace=True)

print('Metadata file read')
print('Building Category, Region and Full Name dictionaries')

category_dict = df_meta.set_index('Project number')['Project category'].to_dict()
region_dict = df_meta.set_index('Project number')['Region'].to_dict()
name_dict = df_meta.set_index('Project number')['Full name'].to_dict()

def get_kind(object_name):
    if 'DELETED' in object_name:
        code = int(object_name.split(',')[1][2:])
        try:
            object_name = material_dict[code]
        except KeyError:
            return 'UNKNOWN'
    return object_name.split(' ')[0].split(',')[0]

print('Remapping values...')

df['Category'] = df['Project number'].map(category_dict)
print('Category map complete')
df['Region'] = df['Project number'].map(region_dict)
print('Region  map complete')
df['Full name'] = df['Project number'].map(name_dict)
print('Project naming  map complete')
df['Material kind'] = df['Material'].apply(get_kind)
print('Material kind complete')

print('Final shape:', df.shape)

path = 'data/clean_v3.csv'

df.to_csv(
    path_or_buf=path,
    index=False
)

print('Dataframe saved to', path)
