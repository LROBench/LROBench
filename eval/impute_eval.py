import pandas as pd 
import os 
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import LogicalImpute
from src.core.enums import OperandType, ImplType
from src.metrics.classic import list_em_acc, df_em_acc
from src.metrics.agent_match import list_agent_acc, df_agent_acc
from src.utils.impute_utils import random_drop


random_seed = 2025


# external knowledge, medium
def pipeline_impute100(args):
    query = "For books published in 2010, impute the missing language name and publisher name based on the book title."
    book = pd.read_csv("./databases/books/book.csv")
    language = pd.read_csv("./databases/books/book_language.csv")
    publisher = pd.read_csv("./databases/books/publisher.csv")
    book = pd.merge(book, language, on='language_id')
    book = pd.merge(book, publisher, on='publisher_id')

    book['publication_date'] = pd.to_datetime(book['publication_date'], errors='coerce')
    book = book[book['publication_date'].dt.year == 2010]
    book['publication_date'] = book['publication_date'].dt.strftime('%Y-%m-%d') 
    # print(book[['language_name', 'publisher_name']])
    book, drop_log = random_drop(book, ['language_name', 'publisher_name'], 0.2, random_seed)
    # print(book[['language_name', 'publisher_name']])
    print(book.columns)

    op = LogicalImpute(operand_type=OperandType.CELL)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells according to the book title", 
                               df = book,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table[['language_name', 'publisher_name']])
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# external knowledge, medium
def pipeline_impute101(args):
    query = "Impute the missing director information of the Disney movies."
    director = pd.read_csv("./databases/disney/director.csv").dropna()
    director, drop_log = random_drop(director, ['director'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the movies", 
                               df = director,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# external knowledge, medium
def pipeline_impute102(args):
    query = "Impute the missing release date, hero, villain, and song information of the characters of Disney movies."
    characters = pd.read_csv("./databases/disney/characters.csv").dropna()
    characters, drop_log = random_drop(characters, ['release_date','hero','villian','song'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the movies", 
                               df = characters,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# external knowledge, hard
def pipeline_impute103(args):
    query = "Impute the missing city and state of the customer information based on their address."
    customers = pd.read_csv("./databases/car_retails/customers.csv")
    customers['addressLine1'] = customers['addressLine1'] + customers['addressLine2']
    customers = customers.drop('addressLine2', axis=1).dropna(subset=['city', 'state'])
    # print(customers.shape)

    customers, drop_log = random_drop(customers, ['city', 'state'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the customer information", 
                               df = customers,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# analogy, easy
def pipeline_impute104(args):
    query = "For songs in the album '1989', impute the missing album name, price, label, and copyright information."
    music = pd.read_csv("./databases/music/amazon_music.csv")
    music = music[music['Album_Name'] == '1989']
    # print(music)
    music, drop_log = random_drop(music, ['Album_Name', 'Price', 'Label', 'Copyright'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the songs", 
                               df = music,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# analogy and external knowledge, easy
def pipeline_impute105(args):
    query = "For animations with 'Conan' in the title, impute the missing producers, type, and year information."
    planet = pd.read_csv("./databases/anime/anime_planet.csv")
    planet = planet[planet['Title'].str.contains('Conan')]
    # print(planet)
    planet, drop_log = random_drop(planet, ['Producers', 'Type', 'Year'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the animations", 
                               df = planet,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table[['Producers', 'Type', 'Year']])
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()



# numerical reasoning, medium
def pipeline_impute106(args):
    query = "For schools with more than 500 students scoring above 1500, impute the missing average SAT score calculated from reading, math, and writing scores."
    satscores = pd.read_csv("./databases/california_schools/satscores.csv")
    satscores = satscores[satscores['NumGE1500'] > 500]
    satscores['AvgScr'] = ((satscores['AvgScrRead'] + satscores['AvgScrMath'] + satscores['AvgScrWrite']) / 3).round(2)
    satscores, drop_log = random_drop(satscores, ['AvgScr'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the score information", 
                               df = satscores,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# numerical reasoning, hard
# The result of LLM is not an EM, with 0.01 discrepancy
def pipeline_impute107(args):
    query = "For transactions with a price greater than 1000, impute the missing unit price based on the total price and amount."
    transactions = pd.read_csv("./databases/debit_card_specializing/transactions_1k.csv")
    transactions = transactions[transactions['Price']>1000]
    transactions['UnitPrice'] = (transactions['Price'] / transactions['Amount']).round(2)
    transactions, drop_log = random_drop(transactions, ['UnitPrice'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the transaction information", 
                               df = transactions,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# numerical reasoning, hard
def pipeline_impute108(args):
    query = "Impute the missing ages of Formula 1 drivers as of January 1, 2025, based on their date of birth."
    def calculate_age(born, today):
        return int(today.year - born.year - ((today.month, today.day) < (born.month, born.day)))
    drivers = pd.read_csv("./databases/formula_1/drivers.csv").dropna()
    drivers['dob'] = pd.to_datetime(drivers['dob'])
    drivers['age'] = drivers['dob'].apply(lambda x: calculate_age(x, pd.to_datetime('2025-01-01')))
    drivers['dob'] = drivers['dob'].dt.strftime('%Y-%m-%d')

    drivers, drop_log = random_drop(drivers, ['age'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing ages of the driver as of January 1, 2025.", 
                               df = drivers,
                               example_num=args.example_num,
                               thinking = args.thinking)
    imputed_table['age'] = imputed_table['age'].astype(int)
    print(drop_log)
    print(imputed_table[['dob','age']])
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# cannot infer, impossible
def pipeline_impute109(args):
    query = "Impute the missing address and phone number of the driver information."
    driver = pd.read_csv("./databases/shipping/driver.csv")
    driver, drop_log = random_drop(driver, ['address', 'phone'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the driver information", 
                               df = driver,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()



# external knowledge, medium
def pipeline_impute110(args):
    query = "For movies directed by Clint Eastwood, impute the missing year and duration information."
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    imdb = imdb[imdb['Director'] == 'Clint Eastwood']
    print(imdb)
    imdb, drop_log = random_drop(imdb, ['Year', 'Duration'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the movie information", 
                               df = imdb,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# external knowledge, medium
def pipeline_impute111(args):
    query = "For movies directed by Christopher Nolan, impute the missing year and duration information."
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    imdb = imdb[imdb['Director'] == 'Christopher Nolan']
    print(imdb)
    imdb, drop_log = random_drop(imdb, ['Year', 'Duration'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the movie information", 
                               df = imdb,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# external knowledge, easy
def pipeline_impute112(args):
    query = "For countries with a population greater than 50 million, impute the missing capital city information."
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    country = country.dropna()
    country = country[country['Population'] > 50000000]
    print(len(country))
    country, drop_log = random_drop(country, ['Capital'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the country information", 
                               df = country,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# external knowledge, easy
def pipeline_impute113(args):
    query = "For cities with a population greater than 500,000, impute the missing state information."
    city = pd.read_csv("./databases/shipping/city.csv")
    city = city[city['population'] > 500000]
    city, drop_log = random_drop(city, ['state'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the city information", 
                               df = city,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# numerical, hard
def pipeline_impute114(args):
    query = "For cities with a population greater than 500,000, impute the missing population density (rounded to 2 decimal places) based on population and area."
    city = pd.read_csv("./databases/shipping/city.csv")
    city = city[city['population'] > 500000]
    city['population_density'] = (city['population'] / city['area']).round(2)
    city, drop_log = random_drop(city, ['population_density'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing population density with 2 decimals", 
                               df = city,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# semantic, medium
def pipeline_impute115(args):
    query = "Impute the missing full book language name."
    language = pd.read_csv("./databases/books/book_language.csv")
    language, drop_log = random_drop(language, ['language_name'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing language full name", 
                               df = language,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# numerical, medium
def pipeline_impute116(args):
    query = "Impute the missing total revenue of Disney movies."
    revenue = pd.read_csv("./databases/disney/revenue.csv")
    revenue = revenue.fillna(0)
    revenue, drop_log = random_drop(revenue, ['Total'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing revenue", 
                               df = revenue,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# numerical, medium
def pipeline_impute117(args):
    query = "For Apple brand electronics, impute the missing discount (rounded to 2 decimal places) based on the original price and Amazon price."
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon.dropna().reset_index(drop=True)
    amazon = amazon[(amazon['Brand'] == 'Apple')]
    amazon['Original_Price'] = amazon['Original_Price'].str.replace(',', '').str.replace('$', '').astype(float)
    amazon['Amazon_Price'] = amazon['Amazon_Price'].str.replace(',', '').str.replace('$', '').astype(float)
    amazon = amazon.sort_values(by='Original_Price', ascending=False)[:30]
    
    amazon['Discount'] = ((amazon['Original_Price'] - amazon['Amazon_Price']) / amazon['Original_Price']).round(2)

    amazon, drop_log = random_drop(amazon, ['Discount'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing discount with 2 decimals", 
                               df = amazon,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# numerical, hard
def pipeline_impute118(args):
    query = "For Dell brand electronics, impute the missing original prices (rounded to 2 decimal places) based on the Amazon price and discount."
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon.dropna().reset_index(drop=True)
    amazon = amazon[(amazon['Brand'] == 'Dell')]
    amazon['Original_Price'] = amazon['Original_Price'].str.replace(',', '').str.replace('$', '').astype(float)
    amazon['Amazon_Price'] = amazon['Amazon_Price'].str.replace(',', '').str.replace('$', '').astype(float)
    amazon = amazon.sort_values(by='Original_Price', ascending=False)[:30]
    
    amazon['Discount'] = ((amazon['Original_Price'] - amazon['Amazon_Price']) / amazon['Original_Price']).round(2)

    amazon, drop_log = random_drop(amazon, ['Original_Price'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing prices with 2 decimals", 
                               df = amazon,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# numerical, hard
def pipeline_impute119(args):
    query = "For Lenovo brand electronics, impute the missing Amazon prices (rounded to 2 decimal places) based on the original price and discount."
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon.dropna().reset_index(drop=True)
    amazon = amazon[(amazon['Brand'] == 'Lenovo')]
    amazon['Original_Price'] = amazon['Original_Price'].str.replace(',', '').str.replace('$', '').astype(float)
    amazon['Amazon_Price'] = amazon['Amazon_Price'].str.replace(',', '').str.replace('$', '').astype(float)
    amazon = amazon.sort_values(by='Original_Price', ascending=False)[:30]
    
    amazon['Discount'] = ((amazon['Original_Price'] - amazon['Amazon_Price']) / amazon['Original_Price']).round(2)

    amazon, drop_log = random_drop(amazon, ['Amazon_Price'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing prices with 2 decimals", 
                               df = amazon,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# external, medium
def pipeline_impute120(args):
    query = "For animations with 'One Piece' in the title, impute the missing episodes and type information."
    anime = pd.read_csv("./databases/anime/anime_planet.csv")
    anime = anime[anime['Title'].str.contains('One Piece')].dropna()

    anime, drop_log = random_drop(anime, ['Episodes', 'Type'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing animation information", 
                               df = anime,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# external, medium
def pipeline_impute121(args):
    query = "For animations with 'Naruto' in the title, impute the missing episodes and type information."
    anime = pd.read_csv("./databases/anime/anime_planet.csv")
    anime = anime[anime['Title'].str.contains('Naruto')].dropna()

    anime, drop_log = random_drop(anime, ['Episodes', 'Type'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing animation information", 
                               df = anime,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# external, medium
def pipeline_impute122(args):
    query = "For songs in the album 'Born This Way', impute the missing release date and duration information."
    music = pd.read_csv("./databases/music/amazon_music.csv")
    music = music[music['Album_Name'] == 'Born This Way']
    print(music)
    music, drop_log = random_drop(music, ['Released', 'Time'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the songs", 
                               df = music,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# external, medium
def pipeline_impute123(args):
    query = "For songs in the album '+', impute the missing release date and duration information."
    music = pd.read_csv("./databases/music/amazon_music.csv")
    music = music[music['Album_Name'] == '+']
    print(music)
    music, drop_log = random_drop(music, ['Released', 'Time'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of the songs", 
                               df = music,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


def pipeline_impute124(args):
    query = "For superheroes whose names start with 'T', impute the missing gender information."
    superhero = pd.read_csv("./databases/superhero/superhero.csv")
    gender = pd.read_csv("./databases/superhero/gender.csv")
    superhero = pd.merge(superhero, gender, left_on='gender_id', right_on='id')
    superhero = superhero.dropna()
    superhero = superhero[superhero['superhero_name'].str.startswith('T')]
    print(superhero)

    superhero, drop_log = random_drop(superhero, ['gender'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of genders", 
                               df = superhero,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# numerical, medium
def pipeline_impute125(args):
    query = "For football players born in 1998, impute the missing BMI values based on their height and weight."
    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player = player[pd.to_datetime(player['birthday']).dt.year == 1998]
    player['bmi'] = (player['weight'] * 0.45359 / (player['height']/100) / (player['height']/100)).round(2).tolist()
    print(len(player))
    player, drop_log = random_drop(player, ['bmi'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of bmis", 
                               df = player,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# numerical, hard
def pipeline_impute126(args):
    query = "For football players born in 1998, impute the missing weight values based on their height and BMI."
    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player = player[pd.to_datetime(player['birthday']).dt.year == 1998]
    player['bmi'] = (player['weight'] * 0.45359 / (player['height']/100) / (player['height']/100)).round(2).tolist()
    print(len(player))
    player, drop_log = random_drop(player, ['weight'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of weights", 
                               df = player,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# numerical, hard
def pipeline_impute127(args):
    query = "For football players born in 1999, impute the missing height values based on their weight and BMI."
    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player = player[pd.to_datetime(player['birthday']).dt.year == 1999]
    player['bmi'] = (player['weight'] * 0.45359 / (player['height']/100) / (player['height']/100)).round(2).tolist()
    print(len(player))
    player, drop_log = random_drop(player, ['height'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells of heights", 
                               df = player,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()

# impossible
def pipeline_impute128(args):
    query = "Impute the missing last name and first name of the car retail employees."
    employees = pd.read_csv("./databases/car_retails/employees.csv")
    employees, drop_log = random_drop(employees, ['lastName','firstName'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells ", 
                               df = employees,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# partially impossible
def pipeline_impute129(args):
    query = "Impute the missing city and phone number of the car retail offices."
    offices = pd.read_csv("./databases/car_retails/offices.csv")
    offices, drop_log = random_drop(offices, ['city', 'phone'], 0.2, random_seed)
    op = LogicalImpute(operand_type=OperandType.CELL)   
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the missing cells ", 
                               df = offices,
                               example_num=args.example_num,
                               thinking = args.thinking)
    print(drop_log)
    print(imputed_table)
    return (df_em_acc(imputed_table, drop_log), df_agent_acc(imputed_table, drop_log)), op.get_tokens()


# external knowledge, easy
def pipeline_impute200(args):
    query = "Impute the gender of the hero for each Disney character."
    truth = [
        'female','male','male','male','male','female','female','male','mixed','female','male',
        'male','male','mixed','male','mixed','male','male','male','female','mixed','female',
        'male','male','female','male','male','female','male','male','male','male','male','male',
        'male','female','female','male','female','male'
    ]
    characters = pd.read_csv("./databases/disney/characters.csv")
    characters = characters.dropna() 
    print(characters.shape)

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the gender of the hero", 
                               df = characters,
                               depend_on = 'hero', 
                               new_col = 'gender',
                               thinking = args.thinking)
    pred = imputed_table['gender'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# external knowledge, medium
def pipeline_impute201(args):
    query = "Impute the birth year of each Disney director who does not have full credits."
    truth = [
        1900, 1895, 1895, 1900, 1909, 1902, 1909, 1909, 1901, 1909, 
        1906, 1901, 1903, 1903, 1901, 1909, 1909, 1909, 1909, 1909, 
        1909, 1909, 1915, 1920, 1953, 1952, 1953, 1954, 1960, 1953, 
        1949, 1954, 1960, 1953, 1958, 1960, 1963, 1960, 1960, 1962, 
        1953, 1968, 1958, 1960, 1970, 1968, 1953, 1975, 1970, 1963, 
        1960, 1969, 1968, 1953
    ]
    director = pd.read_csv("./databases/disney/director.csv")
    director = director[director['director'] != 'full credits']
    print(director.shape)

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the year of birth of each director", 
                               df = director,
                               depend_on = 'director', 
                               new_col = 'birth_year',
                               thinking = args.thinking)
    imputed_table['birth_year'] = imputed_table['birth_year'].map(int)
    pred = imputed_table['birth_year'].tolist()
    print(imputed_table)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# external knowledge, easy
def pipeline_impute202(args):
    query = "Impute the country of each city in the first 100 rows of the address table."
    address = pd.read_csv("./databases/books/address.csv")[:100]
    country = pd.read_csv("./databases/books/country.csv")
    truth = pd.merge(address, country, on='country_id')['country_name'].tolist()
    print(truth)
    address = address.drop('country_id', axis=1)

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the country of the city", 
                               df = address,
                               depend_on = 'city', 
                               new_col = 'country',
                               thinking = args.thinking)
    pred = imputed_table['country'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# external knowledge, medium
def pipeline_impute203(args):
    query = "Impute the publishing company of each superhero based on their name in the first 100 rows of the superhero table."
    superhero = pd.read_csv("./databases/superhero/superhero.csv").dropna(subset=['publisher_id'])[:100]
    publisher = pd.read_csv("./databases/superhero/publisher.csv")
    truth = pd.merge(superhero, publisher, left_on='publisher_id', right_on='id')['publisher_name'].tolist()
    print(truth)
    superhero = superhero.drop('publisher_id', axis=1)

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Impute the publishing company of each superhero.", 
                               df = superhero,
                               depend_on = ['superhero_name', 'full_name'], 
                               new_col = 'publisher',
                               thinking = args.thinking)
    pred = imputed_table['publisher'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# external knowledge and reasoning, hard
# TODO: questionable
def pipeline_impute204(args):
    query = "Assign each movie to one of the following categories based on its title and description: Action, Animation, Children, Classics, Comedy, Documentary, Drama, Family, Foreign, Games, Horror, Music, New, Sci-Fi, Sports, or Travel, in the first 50 rows of the film table."
    film = pd.read_csv("./databases/movie_3/film.csv")[:50]
    mid = pd.read_csv("./databases/movie_3/film_category.csv")
    category = pd.read_csv("./databases/movie_3/category.csv")
    truth = film.merge(mid, on='film_id').merge(category, on='category_id')['name'].tolist()
    print(len(truth))
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Assign each movie to a category: {{Action}}, {{Animation}}, {{Children}}, {{Classics}}, {{Comedy}}, \
                                {{Documentary}}, {{Drama}}, {{Family}}, {{Foreign}}, {{Games}}, {{Horror}}, {{Music}}, {{New}}, {{Sci-Fi}}, {{Sports}}, {{Travel}}", 
                               df = film,
                               depend_on = ['title', 'description'], 
                               new_col = 'category',
                               thinking = args.thinking)
    
    pred = imputed_table['category'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# multi-hop numerical reasoning, hard
def pipeline_impute205(args):
    query = "Calculate the BMI of each football player born in 1997 based on their height in cm and weight in lb."
    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player = player[pd.to_datetime(player['birthday']).dt.year == 1997]
    truth = (player['weight'] * 0.45359 / (player['height']/100) / (player['height']/100)).round(2).tolist()
    print(len(truth))
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Calculate the BMI of each player based on the 'height' in cm and the 'weight' in lb", 
                               df = player,
                               depend_on = ['height', 'weight'], 
                               new_col = 'bmi',
                               thinking = args.thinking)
    imputed_table['bmi'] = imputed_table['bmi'].map(float).round(2)
    pred = imputed_table['bmi'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# numerical reasoning, medium
def pipeline_impute206(args):
    query = "For each order in the first 50 rows of the orders table, calculate the number of days between the shipped date and the order date."
    orders = pd.read_csv("./databases/car_retails/orders.csv")[:50]
    truth = ((pd.to_datetime(orders['shippedDate']) - pd.to_datetime(orders['orderDate'])).dt.days).tolist()
    print(len(truth))
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Calculate the number of days between shippedDate and orderDate", 
                               df = orders,
                               depend_on = ['shippedDate', 'orderDate'], 
                               new_col = 'days',
                               thinking = args.thinking)
    imputed_table['days'] = imputed_table['days'].map(int)
    pred = imputed_table['days'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# numerical reasoning, medium
def pipeline_impute207(args):
    query = "For transactions with a price greater than 1000, calculate the unit price based on the total price and amount."
    transactions = pd.read_csv("./databases/debit_card_specializing/transactions_1k.csv")
    transactions = transactions[transactions['Price']>1000]
    truth = (transactions['Price'] / transactions['Amount']).round(2).tolist()
    print(len(truth))
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Calculate the unit price", 
                               df = transactions,
                               depend_on = ['Amount', 'Price'], 
                               new_col = 'UnitPrice',
                               thinking = args.thinking)
    imputed_table['UnitPrice'] = imputed_table['UnitPrice'].map(float).round(2)
    pred = imputed_table['UnitPrice'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# numerical reasoning, medium
def pipeline_impute208(args):
    query = "For each school in the first 50 rows of the frpm table, calculate the Percent (%) Eligible Free (K-12) based on the enrollment and free meal count."
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")
    frpm = frpm.dropna(subset=['Enrollment (K-12)', 'Free Meal Count (K-12)', 'Percent (%) Eligible Free (K-12)'])[:50]
    truth = (frpm['Percent (%) Eligible Free (K-12)'] * 100).round(2).tolist()
    frpm = frpm.drop('Percent (%) Eligible Free (K-12)', axis = 1)
    print(len(truth))

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Calculate the Percent (%) Eligible Free (K-12) of each school", 
                               df = frpm,
                               depend_on = ['Enrollment (K-12)', 'Free Meal Count (K-12)'], 
                               new_col = 'Percent (%) Eligible Free (K-12)',
                               thinking = args.thinking)
    imputed_table['Percent (%) Eligible Free (K-12)'] = imputed_table['Percent (%) Eligible Free (K-12)'].map(float).round(2)
    pred = imputed_table['Percent (%) Eligible Free (K-12)'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# numerical reasoning, medium
def pipeline_impute209(args):
    query = "For each school in the first 50 rows of the satscores table, calculate the excellent rate (between 0 and 1) based on the number of test takers and the number of students scoring above 1500."
    satscores = pd.read_csv("./databases/california_schools/satscores.csv")
    satscores = satscores.dropna(subset=['NumTstTakr', 'NumGE1500'])[:50]
    truth = (satscores['NumGE1500'] / satscores['NumTstTakr']).round(2).tolist()
    print(len(truth))
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Calculate the excellent rate (0-1) of each school", 
                               df = satscores,
                               depend_on = ['NumTstTakr', 'NumGE1500'], 
                               new_col = 'ExcelRate',
                               thinking = args.thinking)
    imputed_table['ExcelRate'] = imputed_table['ExcelRate'].map(float).round(2)
    pred = imputed_table['ExcelRate'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# external knowledge, easy
def pipeline_impute210(args):
    query = "Impute the year attribute (Leap Year or Common Year) for each date in the 'yearmonth' table."
    truth = ['Leap Year', 'Common Year', 'Common Year', 'Common Year', 'Leap Year', 'Leap Year', 'Common Year', 'Common Year', 'Common Year', 'Common Year', 'Common Year', 'Common Year', 'Common Year', 'Leap Year', 'Leap Year', 'Leap Year', 'Leap Year', 'Leap Year', 'Leap Year', 'Common Year', 'Common Year']

    consumption = pd.read_csv("./databases/debit_card_specializing/yearmonth.csv")
    consumption = consumption[['Date']].drop_duplicates()
    print(consumption)
        
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="Impute the year attribute of the input date. Options are 'Leap Year' or 'Common Year'.", 
                               df = consumption,
                               depend_on = 'Date', 
                               new_col = 'YearAttr', 
                               thinking = args.thinking)
    pred = result['YearAttr'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# external knowledge, easy
def pipeline_impute211(args):
    query = "Impute the Chinese Zodiac sign for each year in the Disney revenue data."
    truth = ['Goat', 'Monkey', 'Rooster', 'Dog', 'Pig', 'Rat', 'Ox', 'Tiger', 'Rabbit', 'Dragon', 'Snake', 
             'Horse', 'Goat', 'Monkey', 'Rooster', 'Dog', 'Pig', 'Rat', 'Ox', 'Tiger', 'Rabbit', 'Dragon', 
             'Snake', 'Horse', 'Goat', 'Monkey']
    disney = pd.read_csv("./databases/disney/revenue.csv")
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="Impute the Chinese Zodiac of each year", 
                               df = disney,
                               depend_on = 'Year', 
                               new_col = 'Zodiac', 
                               thinking = args.thinking)
    pred = result['Zodiac'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# numerical, hard
def pipeline_impute212(args):
    query = "Impute the discount of Apple product with the Top 30 highest original prices based on the original price and Amazon price."
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon.dropna().reset_index(drop=True)
    amazon = amazon[(amazon['Brand'] == 'Apple')]
    amazon['Original_Price'] = amazon['Original_Price'].str.replace(',', '').str.replace('$', '').astype(float)
    amazon['Amazon_Price'] = amazon['Amazon_Price'].str.replace(',', '').str.replace('$', '').astype(float)
    amazon = amazon.sort_values(by='Original_Price', ascending=False)[:30]
    
    truth = ((amazon['Original_Price'] - amazon['Amazon_Price']) / amazon['Original_Price']).round(2).tolist()

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="Impute the discount of product", 
                               df = amazon,
                               depend_on = ['Original_Price', 'Amazon_Price'], 
                               new_col = 'Discount', 
                               thinking = args.thinking)
    pred = result['Discount'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# numerical, medium
def pipeline_impute213(args):
    query = "Impute the population density of each city with a population greater than 500,000 based on population and area."
    city = pd.read_csv("./databases/shipping/city.csv")
    city = city[city['population'] > 500000]
    truth = (city['population'] / city['area']).round(2).tolist()

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="Impute the population density", 
                               df = city,
                               depend_on = ['population', 'area'], 
                               new_col = 'population_density', 
                               thinking = args.thinking)
    pred = result['population_density'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# external knowledge, medium
def pipeline_impute214(args):
    query = "Impute the altitude of Formula 1 circuits located at latitudes above 50 based on the circuit name and location."
    truth = [153, 401, 620, 90, 60, 2, 50, 140, 13, 12, 34]
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    circuits = circuits[circuits['lat'] > 50]

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="Impute the altitude of circuits", 
                               df = circuits,
                               depend_on = ['name', 'location'], 
                               new_col = 'alt', 
                               thinking = args.thinking)
    pred = result['alt'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# external knowledge, medium
def pipeline_impute215(args):
    query = "For the 30 most populous cities, what are the abbreviated state codes?"
    truth = ['NY', 'CA', 'IL', 'TX', 'PA', 'AZ', 'CA', 'TX', 'TX', 'MI', 'CA', 'IN', 'CA', 'FL', 'OH', 'TX', 'MD', 'TN', 'WI', 'MA', 'DC', 'TX', 'WA', 'CO', 'TN', 'NC', 'TX', 'OR', 'OK', 'AZ']
    city = pd.read_csv("./databases/shipping/city.csv")
    city = city.sort_values(by='population', ascending=False)[:30]

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="Impute the state abbreviated code", 
                               df = city,
                               depend_on = ['state'], 
                               new_col = 'state_code', 
                               thinking = args.thinking)
    pred = result['state_code'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# external knowledge, medium
def pipeline_impute216(args):
    query = "For each office location, who was the country's top leader in 2023?"
    truth = ['Joe Biden', 'Joe Biden', 'Joe Biden', 'Emmanuel Macron', 'Fumio Kishida', 'Anthony Albanese', 'Rishi Sunak']
    offices = pd.read_csv("./databases/car_retails/offices.csv")

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="Impute the name of the country's top leader in 2023", 
                               df = offices,
                               depend_on = ['country'], 
                               new_col = 'leader', 
                               thinking = args.thinking)
    pred = result['leader'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# fuzzy reformatting, easy
def pipeline_impute217(args):
    query = "For customers whose first name starts with 'V', what is the institution (domain) of their email address?"
    truth = ['e-recht24.de', 'usnews.com', 'archive.org', 'Mailchimp', 'upenn.edu', 'networksolutions.com', 'blinklist.com', 'shinystat.com', 'comsenz.com', 'cmu.edu', 'i2i.jp', 'patch.com', 'livejournal.com', 'flickr.com', 'seesaa.net', 'elegantthemes.com', 'slideshare.net', 'bigcartel.com', 'feedburner.com', 'prweb.com', 'lycos.com', 'Google', 'blogger.com', 'nationalgeographic.com', 'xrea.com', 'zimbio.com', 'vk.com', 'posterous.com', 'hp.com', 'elegantthemes.com', 'guardian.co.uk', 'ucla.edu', 'cbslocal.com', 'NetEase', 'oracle.com', 'epa.gov', 'bbb.org']
    customer = pd.read_csv("./databases/books/customer.csv")
    customer = customer[customer['first_name'].str.startswith('V')]
    print(customer)
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="Impute the institution of the email", 
                               df = customer,
                               depend_on = ['email'], 
                               new_col = 'email_server', 
                               thinking = args.thinking)
    pred = result['email_server'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# fuzzy reformatting, easy
def pipeline_impute218(args):
    query = "For customers whose first name starts with 'A', reformat their first purchase date into mm/dd/yyyy format."
    truth = ['10/26/2012', '12/06/2012', '01/13/2013', '08/01/2013', '07/09/2012', '03/23/2013', 
             '11/03/2012', '07/27/2013', '01/14/2013', '01/08/2013', '09/05/2012', '01/12/2013', 
             '08/29/2012', '03/07/2013', '12/08/2012', '06/07/2013', '02/19/2013', '10/21/2012', 
             '11/28/2012', '04/03/2013', '04/25/2013', '10/27/2012', '01/22/2013', '08/29/2013', 
             '08/24/2012', '07/19/2012', '04/13/2013', '10/08/2012', '03/09/2013', '07/22/2013', 
             '04/22/2013', '08/21/2013', '08/13/2012', '05/31/2013', '04/17/2013', '08/20/2012',
             '06/01/2013', '05/22/2013', '07/09/2012', '03/10/2013', '02/23/2013', '04/21/2013']
    customers = pd.read_csv("./databases/beer_factory/customers.csv")
    customers = customers[customers['First'].str.startswith('A')]
    print(len(customers))
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="reformat the date into mm/dd/yyyy", 
                               df = customers,
                               depend_on = ['FirstPurchaseDate'], 
                               new_col = 'newDate', 
                               thinking = args.thinking)
    pred = result['newDate'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# external knowledge, easy
def pipeline_impute219(args):
    query = "For each Olympic host city, what country does the city belong to?"
    truth = ['Spain', 'United Kingdom', 'Belgium', 'France', 'Canada', 'France', 'Norway', 'United States', 'United States', 'Finland', 'United States', 'Australia', 'United States', 'Sweden', 'Russia', 'Japan', 'Italy', 'China', 'Brazil', 'Greece', 'United States', 'Austria', 'Bosnia and Herzegovina', 'Mexico', 'Germany', 'South Korea', 'Germany', 'Norway', 'Italy', 'Australia', 'Italy', 'Netherlands', 'Canada', 'Russia', 'Japan', 'Canada', 'France', 'Japan', 'France', 'United States', 'Switzerland', 'Germany']
    city = pd.read_csv("./databases/olympics/city.csv")
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="impute the country of the city", 
                               df = city,
                               depend_on = ['city_name'], 
                               new_col = 'country', 
                               thinking = args.thinking)
    pred = result['country'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

    
# external knowledge, medium
def pipeline_impute220(args):
    query = "For each Olympic sport, what was the first year it was included in the Olympics?"
    truth = [float('nan'), 1936, 1924, 1900, 1912, 1896, 1992, 1992, 1936, 1900, 1996, 1960, 1924, 1904, 1936, 1900, 1900, 1924, 1998, 1896, 1904, 1900, 1896, 1908, 1900, 1992, 1900, 1896, 1936, 1908, 1920, 1908, 1964, 1904, 1964, 1924, 1912, 1908, 1924, 1900, 1908, 1984, 1904, 1900, 1900, 2016, 1900, 1896, 1992, 1928, 1924, 1998, 1996, 1924, 1896, 1984, 1988, 2000, 1896, 2000, 2000, 1900, 1964, 1900, 1896, 1896]
    sport = pd.read_csv("./databases/olympics/sport.csv")
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                               condition="impute the first year of the sport's inclusion in the Olympics", 
                               df = sport,
                               depend_on = ['sport_name'], 
                               new_col = 'year', 
                               thinking = args.thinking)
    pred = result['year'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# external knowledge, medium
def pipeline_impute221(args):
    query = "For each region, what is the abbreviated currency code used in that region?"
    truth = ['AFN', 'ANG', 'ALL', 'DZD', 'EUR', 'AOA', 'XCD', 'AUD', 'ARS', 'AMD', 'AWG', 'USD', 'AUD', 'EUR', 'AZN', 'BSD', 'BDT', 'BBD', 'BIF', 'EUR', 'XOF', 'BMD', 'BTN', 'BAM', 'BZD', 'BYN', 'CZK', 'BOB', 'BWP', 'BRL', 'BHD', 'BND', 'BGN', 'XOF', 'XAF', 'KHR', 'CAD', 'KYD', 'XAF', 'XAF', 'CLP', 'CNY', 'XOF', 'XAF', 'CDF', 'NZD', 'COP', 'KMF', 'CVE', 'CRC', 'EUR', 'EUR', 'CUP', 'EUR', 'CZK', 'DKK', 'DJF', 'XCD', 'DOP', 'USD', 'EGP', 'ERN', 'USD', 'EUR', 'EUR', 'ETB', 'SUR', 'FJD', 'EUR', 'EUR', 'DEM', 'USD', 'XAF', 'GMD', 'GBP', 'XOF', 'M', 'GEL', 'XAF', 'EUR', 'GHS', 'EUR', 'XCD', 'GTQ', 'GNF', 'USD', 'GYD', 'HTG', 'HKD', 'HNL', 'HUF', 'IDR', 'INR', 'N/A', 'IRR', 'EUR', 'IQD', 'ISK', 'ILS', 'USD', 'EUR', 'USD', 'JMD', 'JOD', 'JPY', 'KZT', 'KES', 'KGS', 'AUD', 'KRW', 'EUR', 'SAR', 'KWD', 'LAK', 'EUR', 'LYD', 'LRD', 'XCD', 'LSL', 'LBP', 'CHF', 'EUR', 'EUR', 'MGA', 'MYR', 'MAD', 'MYR', 'MWK', 'MDL', 'MVR', 'MXN', 'MNT', 'USD', 'MKD', 'XOF', 'EUR', 'EUR', 'EUR', 'MZN', 'MUR', 'MRU', 'MMK', 'NAD', 'MYR', 'NIO', 'EUR', 'NPR', 'CAD', 'NGN', 'XOF', 'NOK', 'AUD', 'NZD', 'OMR', 'PKR', 'PAB', 'PYG', 'PEN', 'PHP', 'ILS', 'USD', 'PGK', 'PLN', 'EUR', 'KPW', 'USD', 'QAR', 'RHD', 'USD', 'RON', 'ZAR', 'RUB', 'RWF', 'EUR', 'WST', 'YUM', 'XOF', 'SCR', 'SGD', 'USD', 'SLL', 'EUR', 'EUR', 'SBD', 'SOS', 'RSD', 'LKR', 'SSP', 'STN', 'SDG', 'CHF', 'SRD', 'EUR', 'SEK', 'SZL', 'SYP', 'TZS', 'CSK', 'TOP', 'THB', 'TJS', 'TMT', 'USD', 'XOF', 'TWD', 'TTD', 'TND', 'TRY', 'AUD', 'AED', 'EGP', 'UGX', 'UAH', 'XXX', 'RUB', 'UYU', 'USD', 'UZS', 'VUV', 'VES', 'VND', 'XCD', 'P', 'XCD', 'YER', 'YER', 'YDD', 'YUD', 'ZMW', 'ZWL', 'SGD']
    region = pd.read_csv("./databases/olympics/noc_region.csv")
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    region = op.execute(impl_type=args.impl, 
                condition="impute the abbreviated currency name of the region", 
                df = region,
                depend_on = ['region_name'], 
                new_col = 'currency', 
                thinking = args.thinking)
    pred = region['currency'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# numerical, medium
def pipeline_impute222(args):
    query = "For each record in the Disney revenue table, what is the ratio of Studio Entertainment revenue to total revenue?"
    revenue =  pd.read_csv("./databases/disney/revenue.csv")
    revenue = revenue.dropna(subset = ['Studio Entertainment[NI 1]', 'Total'])
    truth = (revenue['Studio Entertainment[NI 1]'] / revenue['Total']).tolist()
    op = LogicalImpute(OperandType.COLUMN)
    revenue = op.execute(impl_type=args.impl, 
                condition = "Calculate the ratio of Studio Entertainment revenue to total revenue",
                df = revenue, 
                depend_on = ['Studio Entertainment[NI 1]', 'Total'],
                new_col = 'ratio',
                thinking=args.thinking)
    pred = revenue['ratio'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# numerical, medium
def pipeline_impute223(args):
    query = "For each budget entry, what proportion of the allocated amount has already been spent?"
    budget = pd.read_csv("./databases/student_club/budget.csv")
    truth = (budget['spent']/budget['amount']).tolist()
    op = LogicalImpute(OperandType.COLUMN)
    budget = op.execute(impl_type=args.impl, 
                condition="impute the proportion of money already spent", 
                df = budget,
                depend_on = ['spent' ,'remaining', 'amount'], 
                new_col = 'propotion', 
                thinking = args.thinking)
    pred = budget['propotion'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# external knowledge, easy
def pipeline_impute224(args):
    query = "For each student club event, what season does the event take place in?"
    truth = ['winter', 'winter', 'autumn', 'spring', 'Fall', 'spring', 'autumn', 'spring', 'autumn', 'fall', 'winter', 'fall', 'autumn', 'autumn', 'spring', 'autumn', 'Spring', 'spring', 'autumn', 'Spring', 'spring', 'winter', 'winter', 'autumn', 'spring', 'winter', 'autumn', 'autumn', 'fall', 'autumn', 'Winter', 'fall', 'Spring', 'winter', 'autumn', 'Spring', 'winter', 'autumn', 'spring', 'spring', 'fall', 'autumn']
    event = pd.read_csv("./databases/student_club/event.csv")
    op = LogicalImpute(OperandType.COLUMN)
    event = op.execute(impl_type=args.impl, 
                condition="impute the season of the event", 
                df = event,
                depend_on = ['event_name', 'event_date'], 
                new_col = 'season', 
                thinking = args.thinking)
    pred = event['season'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

# external knowledge, medium
def pipeline_impute225(args):
    query = "For each root beer brand, what is the UTC time zone of the city where it is located?"
    truth = ['UTC-8', 'UTC-5', 'UTC-5', 'UTC-6', 'UTC-8', 'UTC-8', 'UTC+10:00', 'UTC-8', 'UTC-6', 'UTC-6', 'UTC-5', 'UTC-6', 'UTC-8', 'UTC-6', 'UTC-8', 'UTC-6', 'UTC-8', 'UTC-8', 'UTC-5', 'UTC-8', 'UTC-5', 'UTC-8', 'UTC-5', 'UTC-6']    
    brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    op = LogicalImpute(OperandType.COLUMN)
    brand = op.execute(impl_type=args.impl, 
                condition="Impute the UTC time zone of the city", 
                df = brand,
                depend_on = ['City', 'Country'], 
                new_col = 'TimeZone', 
                thinking = args.thinking)
    pred = brand['TimeZone'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# numerical, easy
def pipeline_impute226(args):
    query = "For the dishes with the Top 30 highest prices, what is the time span between when the dish first appeared and last appeared?"
    dish = pd.read_csv("./databases/menu/Dish.csv")
    dish.dropna(subset = ['first_appeared', 'last_appeared'])
    dish = dish.sort_values('highest_price', ascending=False)[:30]
    print(dish)
    truth = (dish['last_appeared'] - dish['first_appeared']).tolist()
    op = LogicalImpute(OperandType.COLUMN)
    dish =  op.execute(impl_type=args.impl, 
                condition="Impute the time span of its duration", 
                df = dish,
                depend_on = ['first_appeared', 'last_appeared'], 
                new_col = 'span', 
                thinking = args.thinking)
    pred = dish['span'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# external, medium
def pipeline_impute227(args):
    query = "For each desert, which continent is the main part of the desert located in?"
    truth = ['Asia', 'South America', 'Africa', 'North America', 'North America', 'North America', 'Africa', 'Asia', 'Asia', 'Asia', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Australia', 'Asia', 'Africa', 'Africa', 'North America', 'North America', 'Australia', 'Australia', 'Asia', 'Africa', 'Africa', 'Africa', 'Asia', 'Asia', 'Asia', 'Africa', 'North America', 'Asia', 'Africa', 'Asia', 'Asia', 'Africa', 'Australia', 'Asia', 'North America', 'Asia', 'Australia', 'Asia', 'Asia', 'Asia', 'Africa', 'Australia', 'North America', 'Australia', 'Asia', 'Asia', 'Africa', 'Australia', 'Africa', 'Africa', 'Asia', 'Africa', 'Asia']
    desert = pd.read_csv("./databases/mondial_geo/desert.csv")
    op = LogicalImpute(OperandType.COLUMN)
    desert =  op.execute(impl_type=args.impl, 
                condition="Impute continent where the main part of the desert is in", 
                df = desert,
                depend_on = ['Name'], 
                new_col = 'Continent', 
                thinking = args.thinking)
    pred = desert['Continent'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# external, medium
def pipeline_impute228(args):
    query = "For each sea, what is its area (in square kilometers)?"
    truth = [797700.0, 3862000.0, 14000000.0, 106460000.0, 377000.0, 1424000.0, 2172000.0, 2000000.0, 436000.0, 2754000.0, 1249000.0, 220000.0, 2172000.0, 1550000.0, 181000.0, 70560000.0, 46000.0, 22000.0, 65000.0, 2500000.0, 570000.0, 1383000.0, 168723000.0, 251000.0, 438000.0, 39000.0, 978000.0, 1583000.0, 987000.0, 32000.0, 3500000.0, 280000.0, 290000.0, 75000.0, 380000.0]
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    op = LogicalImpute(OperandType.COLUMN)
    sea =  op.execute(impl_type=args.impl, 
                condition="Impute the sea area", 
                df = sea,
                depend_on = ['Name'], 
                new_col = 'Area', 
                thinking = args.thinking)
    pred = sea['Area'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# numerical, hard
def pipeline_impute229(args):
    query = "For the 30 most recent transactions, what is the price converted to US dollars?"
    truth = ['5.90', '5.78', '3.69', '1.78', '27.58', '12.99', '9.80', '12.02', '33.96', '22.41', '21.03', '13.76', '25.18', '28.58', '8.57', '30.40', '31.69', '19.12', '1.27', '3.21', '12.62', '49.10', '6.94', '4.61', '0.20', '5.89', '18.79', '34.83', '31.34', '14.04']
    customers = pd.read_csv("./databases/debit_card_specializing/customers.csv")
    transactions = pd.read_csv("./databases/debit_card_specializing/transactions_1k.csv")
    merged = pd.merge(transactions, customers, on ='CustomerID')
    merged['Time'] = merged['Date'] + merged['Time']
    merged = merged.sort_values('Time', ascending=False)[:30]
    op = LogicalImpute(OperandType.COLUMN)
    merged =  op.execute(impl_type=args.impl, 
                condition="Impute the price in US dollar", 
                df = merged,
                depend_on = ['Currency', 'Price'], 
                new_col = 'PriceUSD', 
                thinking = args.thinking)
    pred = merged['PriceUSD'].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


# row, table
def pipeline_impute300(args):
    query = "Insert a row for the Disney movie 'Coco' into the characters table."
    truth = ['Coco', '22-Nov-17', 'Miguel', 'Ernesto de la Cruz', 'Remember Me']
    movies = pd.read_csv("./databases/disney/characters.csv")

    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a disney movie named 'coco'", 
                               df = movies,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    # print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute301(args):
    query = "Insert a row for the Disney movie 'Inside Out' into the characters table."
    truth = ['Inside Out', '19-Jun-15', 'Joy', 'Sadness', 'Bundle of Joy']
    movies = pd.read_csv("./databases/disney/characters.csv")
# 'result': [{'movie_title': 'Inside Out', 'release_date': '19-Jun-15', 'hero': 'Riley', 'villian': 'Sadness', 'song': 'Bundle of Joy'}]}
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a disney movie named 'Inside Out'", 
                               df = movies,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()

def pipeline_impute302(args):
    query = "Insert a row for the song 'Anti-Hero' by Taylor Swift into the itunes table."
    truth = [6908, 'Midnights', 'Taylor Swift', '℗ 2022 Republic Records', 'Pop', '21-Oct-22', 'Anti-Hero', '3:20']
    music = pd.read_csv("./databases/music/itunes.csv")
    music = music.drop(['Album_Price', 'Customer_Rating', 'Price'], axis=1)
    
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a song named 'Anti-Hero' by Taylor Swift", 
                               df = music,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    # print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute303(args):
    query = "Insert a row for the song 'Levitating' by Dua Lipa into the itunes table."
    truth = [6908, 'Future Nostalgia', 'Dua Lipa', '℗ 2020 Warner Records', 'Pop,Disco', '27-Mar-20', 'Levitating', '3:23']
    music = pd.read_csv("./databases/music/itunes.csv")
    music = music.drop(['Album_Price', 'Customer_Rating', 'Price'], axis=1)
    
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a song named 'Levitating' by Dua Lipa", 
                               df = music,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    # print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute304(args):
    query = "Insert a row for the superhero 'Luffy' into the merged table of the superhero, colour, and publisher."
    truth = [757, 'Luffy', 'Black', 'Black', 'Yellow', 'Shueisha']
    superhero = pd.read_csv("./databases/superhero/superhero.csv")
    colour = pd.read_csv("./databases/superhero/colour.csv")
    publisher = pd.read_csv("./databases/superhero/publisher.csv")
    superhero = superhero.merge(colour.rename(columns={'id': 'eye_colour_id', 'colour': 'eye_colour'}), on='eye_colour_id')
    superhero = superhero.merge(colour.rename(columns={'id': 'hair_colour_id', 'colour': 'hair_colour'}), on='hair_colour_id')
    superhero = superhero.merge(colour.rename(columns={'id': 'skin_colour_id', 'colour': 'skin_colour'}), on='skin_colour_id')
    superhero = superhero.merge(publisher.rename(columns={'id': 'publisher_id', 'publisher_name': 'publisher'}), on='publisher_id')
    superhero = superhero[['id','superhero_name','eye_colour','hair_colour','skin_colour', 'publisher']]

    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a superhero named 'Luffy'", 
                               df = superhero,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    # print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute305(args):
    query = "Insert a row for the superhero 'Shang-Chi' into the merged table of the superhero, colour, and publisher."
    truth = [757, 'Shang-Chi', 'Brown', 'Black', 'Yellow', 'Marvel Comics']
    superhero = pd.read_csv("./databases/superhero/superhero.csv")
    colour = pd.read_csv("./databases/superhero/colour.csv")
    publisher = pd.read_csv("./databases/superhero/publisher.csv")
    superhero = superhero.merge(colour.rename(columns={'id': 'eye_colour_id', 'colour': 'eye_colour'}), on='eye_colour_id')
    superhero = superhero.merge(colour.rename(columns={'id': 'hair_colour_id', 'colour': 'hair_colour'}), on='hair_colour_id')
    superhero = superhero.merge(colour.rename(columns={'id': 'skin_colour_id', 'colour': 'skin_colour'}), on='skin_colour_id')
    superhero = superhero.merge(publisher.rename(columns={'id': 'publisher_id', 'publisher_name': 'publisher'}), on='publisher_id')
    superhero = superhero[['id','superhero_name','eye_colour','hair_colour','skin_colour', 'publisher']]

    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a superhero named 'Shang-Chi'", 
                               df = superhero,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute306(args):
    query = "Insert a row for the Disney movie 'Elemental' into the characters table."
    truth = ['Elemental', '16-Jun-23', 'Ember Lumen', 'Water Element', 'Steal the Show']
    movies = pd.read_csv("./databases/disney/characters.csv")

    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a disney movie named 'Elemental'", 
                               df = movies,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute307(args):
    query = "Insert a row for the song 'Anti-Hero' by Taylor Swift into the itunes table."
    truth = [6908, 'Midnights', 'Taylor Swift', '\x89ãÑ 2022 Republic Records', 'Pop,Music', '21-Oct-22', 'Anti-hero', '3:20']
    music = pd.read_csv("./databases/music/itunes.csv")
    music = music.drop(['Album_Price', 'Customer_Rating', 'Price'], axis=1)
    
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a song named 'Anti-Hero' by Taylor Swift", 
                               df = music,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute308(args):
    query = "Insert a row for the song 'Lavender Haze' by Taylor Swift into the itunes table."
    truth = [6908, 'Midnights', 'Taylor Swift', '\x89ãÑ 2022 Republic Records', 'Pop,Music', '21-Oct-22', 'Lavender Haze', '3:22']

    music = pd.read_csv("./databases/music/itunes.csv")
    music = music.drop(['Album_Price', 'Customer_Rating', 'Price'], axis=1)
    
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a song named 'Lavender Haze' by Taylor Swift", 
                               df = music,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute309(args):
    query = "Insert a row for the lake 'West Lake' located in China into the lake table."
    truth = ['West Lake', 6.5, 2.8, 7.0, 'freshwater', 'Qiantang River', 120.1536, 30.2414]
    lake = pd.read_csv("./databases/mondial_geo/lake.csv")
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a lake named 'West Lake' in China", 
                               df = lake,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute310(args):
    query = "Insert a row for the lake 'Tai Lake' located in China into the lake table."
    truth = ['Tai Lake', 2338, 4.8, 3.3, 'freshwater', 'Tiao River and Jing River', 120.2167, 31.1667]
    lake = pd.read_csv("./databases/mondial_geo/lake.csv")
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert a lake named 'Tai Lake' in China", 
                               df = lake,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute311(args):
    query = "Insert a row for the Chinese athlete 'Zheng Qinwen' into the person table."
    truth = [135572, 'Zheng Qinwen', 'F', 178, 70]
    person = pd.read_csv("./databases/olympics/person.csv")
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert an athlete named 'Zheng Qinwen' in China", 
                               df = person,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute312(args):
    query = "Insert a row for the Chinese athlete 'Wang Chuqin' into the person table."
    truth = [135572, 'Wang Chuqin', 'M', 182, 67]
    person = pd.read_csv("./databases/olympics/person.csv")
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert an athlete named 'Wang Chuqin' in China", 
                               df = person,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute313(args):
    query = "Insert a row for the city of Vancouver in Canada into the city table."
    truth = [701, 'Vancouver', 'British Columbia', 662248, 123.63]
    city = pd.read_csv("./databases/shipping/city.csv")
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert Vancouver city in Canada", 
                               df = city,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute314(args):
    query = "Insert a row for the city of Toronto in Canada into the city table."
    truth = [701, 'Toronto', 'Ontario', 2794356, 631.10]
    city = pd.read_csv("./databases/shipping/city.csv")
    op = LogicalImpute(operand_type=OperandType.ROW)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Insert Toronto city in Canada", 
                               df = city,
                               example_num=args.example_num,
                               thinking = args.thinking)
    pred = imputed_table.iloc[-1].tolist()
    print(pred)
    return (list_em_acc(truth, pred), list_agent_acc(truth, pred)), op.get_tokens()


def pipeline_impute400(args):
    query = "Create a Disney movie table with columns for movie name, year, and protagonist, containing movies released after 2020."
    op = LogicalImpute(operand_type=OperandType.TABLE)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Create a Disney movie table containing movies released after 2020", 
                               col_names = ['Movie name', 'Year', 'Protagonist'])
    return imputed_table

def pipeline_impute401(args):
    query = "Create a superhero table with columns for id, superhero name, eye colour, hair colour, and skin colour, containing Marvel Comics superheroes introduced after 2020."
    op = LogicalImpute(operand_type=OperandType.TABLE)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Create a Superhero table containing superheroes released after 2020 created by Marvel Comics", 
                               col_names = ['id','superhero_name','eye_colour','hair_colour','skin_colour'])
    return imputed_table

def pipeline_impute402(args):
    query = "Create an animation table with columns for id, title, episodes, year, and protagonist, containing animations produced by Sunrise and released after 2020."
    op = LogicalImpute(operand_type=OperandType.TABLE)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Create an Animation table containing animations released after 2020 produced by Sunrise", 
                               col_names = ['id','Title','Episodes', 'Year', 'Protagonist'])
    return imputed_table

def pipeline_impute403(args):
    query = "Create a song table with columns for id, song name, album name, and singer, containing Taylor Swift's songs released after 2020."
    op = LogicalImpute(operand_type=OperandType.TABLE)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Create a Song table containing songs by Taylor Swift released after 2020", 
                               col_names = ['id','song_name','album_name', 'singer'])
    return imputed_table

def pipeline_impute404(args):
    query = "Create a song table with columns for id, song name, album name, and singer, containing Dua Lipa's songs released after 2020."
    op = LogicalImpute(operand_type=OperandType.TABLE)
    imputed_table = op.execute(impl_type=args.impl, 
                               condition="Create a Song table containing songs by Dua Lipa released after 2020", 
                               col_names = ['id','song_name','album_name', 'singer'])
    return imputed_table




if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(impl=ImplType.LLM_ALL, example_num=3, thinking=False)
    print(pipeline_impute314(args))