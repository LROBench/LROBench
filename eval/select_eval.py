import pandas as pd 
import os 
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import LogicalSelect
from src.core.enums import OperandType, ImplType
from src.metrics.classic import f1_score

# knowl: easy
# Numeric Reasoning and external knowledge
def pipeline_select100(args):
    query = "List all Formula 1 drivers born after 1990-01-20 who are younger than Alex Albon."
    truth = pd.DataFrame([
        ('Esteban',    'Ocon'),
        ('Max',  'Verstappen'),
        ('Lance',    'Stroll')
    ])
    
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")
    drivers['dob'] = pd.to_datetime(drivers['dob'])
    drivers = drivers[drivers['dob'] > '1990-01-20']   
    drivers['dob'] = drivers['dob'].dt.strftime('%Y-%m-%d')
    drivers = drivers[['forename', 'surname', 'dob']]
    # print(drivers.shape)
    print(drivers)

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "The provided birth date is later than the birth date of Alex Albon.",
                        df = drivers, 
                        depend_on = ['forename', 'surname', 'dob'],
                        thinking = args.thinking)
    result = result[['forename', 'surname']]
    print(result)
    return f1_score(truth, result), op.get_tokens()

# knowl: medium
# Semantic Filter
def pipeline_select101(args):
    query = "Among the top 10 most popular tags, which ones are related to statistics?"
    truth = pd.DataFrame({'TagName': ['regression', 'time-series', 'probability', 'hypothesis-testing', 'distributions', 'correlation']})
    
    tags = pd.read_csv("./databases/codebase_community/tags.csv")
    tags = tags.sort_values(by='Count', ascending=False)[:10]
    tags = tags[['TagName']]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "The tag is related to statistics.",
                        df = tags, 
                        depend_on = ['TagName'],
                        thinking = args.thinking)
    
    return f1_score(truth, result), op.get_tokens()

# knowl: easy
# Knowledge-based Filter
def pipeline_select102(args):
    query = "List all counties in the San Francisco Bay Area."
    truth = pd.DataFrame({'County': ['Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco', 'San Mateo', 'Santa Clara', 'Solano', 'Sonoma']})
    
    schools = pd.read_csv("./databases/california_schools/schools.csv")
    schools = schools[['County']]
    schools = schools.drop_duplicates()
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This county is in the San Francisco Bay Area.",
                        df = schools, 
                        depend_on = ['County'],
                        thinking = args.thinking)
    
    return f1_score(truth, result), op.get_tokens()

# knowl: medium
# Knowledge-based Filter
def pipeline_select103(args):
    query = "List the Formula 1 drivers who were born after 1980 and have won the world championship."
    truth = pd.DataFrame({'forename': ['Lewis', 'Nico', 'Fernando', 'Sebastian', 'Max'], 'surname':['Hamilton', 'Rosberg', 'Alonso', 'Vettel', 'Verstappen']})
    
    drivers = pd.read_csv("./databases/formula_1/drivers.csv").dropna()
    drivers = drivers[pd.to_datetime(drivers['dob']).dt.year > 1980]
    drivers = drivers[['forename', 'surname']]
    print(drivers)
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This driver is a Formula 1 world champion.",
                        df = drivers, 
                        depend_on = ['forename', 'surname'],
                        thinking = args.thinking)
    
    return f1_score(truth, result), op.get_tokens()

# knowl: easy
# Fuzzy Filter
def pipeline_select104(args):
    query = "Among the Top 10 most popular posts, select the IDs for posts whose body contains website links."
    truth = pd.DataFrame({'Id': ['6', '423', '138', '20523', '13314', '16921']})
    
    posts = pd.read_csv("./databases/codebase_community/posts.csv")
    posts = posts.sort_values(by='FavoriteCount', ascending=False)[:10]
    posts = posts[['Id', 'Body']]
        
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This post contains websites.",
                        df = posts, 
                        depend_on = ['Body'],
                        thinking = args.thinking)
    result = result[['Id']].astype(str)
    return f1_score(truth, result), op.get_tokens()

# knowl: medium
# Knowledge-based Filter
def pipeline_select105(args):
    query = "List all Formula 1 circuits located in East Asia."
    truth = pd.DataFrame({'name': ['Fuji Speedway', 'Shanghai International Circuit', 'Suzuka Circuit', 'Okayama International Circuit', 'Korean International Circuit']})

    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    circuits = circuits[['name', 'country']]
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This circuits locates in East Asia.",
                        df = circuits, 
                        depend_on = ['country'],
                        thinking = args.thinking)
    result = result[['name']]
    return f1_score(truth, result), op.get_tokens()


# knowl: 
# Fuzzy Filter
def pipeline_select106(args):
    query = "List the IDs of Sony laptop charger products sold on Amazon."
    truth = pd.DataFrame({'ID': ['758', '2280', '2543', '3253', '3790', '3846', '3872', '4020']})
    
    products = pd.read_csv("./databases/electronics/amazon.csv")
    products = products[products['Brand'] == 'Sony']

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This product is a laptop charger.",
                        df = products, 
                        depend_on = ['Name'],
                        thinking = args.thinking)
    result = result[['ID']].astype(str)
    return f1_score(truth, result), op.get_tokens()

# Semantic Filter
def pipeline_select107(args):
    query = "List the IDs of books about natural science published between January 2000 and February 2000."
    truth = pd.DataFrame({'Id': ['960', '8036', '9538']})

    books = pd.read_csv("./databases/books/book.csv")
    
    books['publication_date'] = pd.to_datetime(books['publication_date'])
    books = books[(books['publication_date'] > '2000-01-01') & (books['publication_date'] < '2000-02-29')]
    books['publication_date'] = books['publication_date'].dt.strftime('%Y-%m-%d')

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This book is about natural science.",
                        df = books, 
                        depend_on = ['title'],
                        thinking = args.thinking)
    result = result[['book_id']].astype(str)
    return f1_score(truth, result), op.get_tokens()


# Semantic Filter
def pipeline_select108(args):
    query = "List the IDs of songs by The Chainsmokers that are about love."
    truth = pd.DataFrame({'Sno': ['2586', '2589', '2593']})
    
    music = pd.read_csv("./databases/music/itunes.csv")
    music = music[music['Artist_Name'] == 'The Chainsmokers']
    music = music[['Sno','Artist_Name', 'Song_Name']]
    music.to_csv('output.csv')
        
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This music is about love.",
                        df = music, 
                        depend_on = ['Artist_Name', 'Song_Name'],
                        thinking = args.thinking)
    result = result[['Sno']].astype(str)
    return f1_score(truth, result), op.get_tokens()

# Knowledge-based Filter
def pipeline_select109(args):
    query = "Among the top 10 tallest football players, select the player ids of those who are European."
    truth = pd.DataFrame({'player_api_id': ['148325', '96465', '150209', '27372', '103428', '38567', '543021', '41129', '26585']})
    
    players = pd.read_csv("./databases/european_football_2/Player.csv")
    players = players.sort_values(by='height', ascending=False)[:10]
    players.to_csv("output.csv")

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This player is European.",
                        df = players, 
                        depend_on = ['player_name'],
                        thinking = args.thinking)
    result = result[['player_api_id']].astype(str)
    return f1_score(truth, result), op.get_tokens()


# hard
# hybrid reasoning
def pipeline_select110(args):
    query = "Which customers have shipments in June 2016 where the customer's state is geographically adjacent to the shipment's state?"
    truth = pd.DataFrame(['Eppies Discount Tire & Auto Centers', 
                          'Kenner Welding Inc', 
                          'Sunguard Window Tinting & Truck Accessories', 
                          'R V City', 
                          'Great Dane Trailers Inc'])
    
    shipment = pd.read_csv("./databases/shipping/shipment.csv")
    shipment = shipment[(pd.to_datetime(shipment['ship_date']).dt.month == 6) & (pd.to_datetime(shipment['ship_date']).dt.year == 2016)]
    city = pd.read_csv("./databases/shipping/city.csv")
    customer = pd.read_csv("./databases/shipping/customer.csv")
    shipment = pd.merge(shipment, city, on='city_id')
    shipment = pd.merge(shipment, customer, on='cust_id')
    # print(shipment.columns)
    op = LogicalSelect(operand_type=OperandType.ROW)
    df = op.execute(impl_type=args.impl, 
                    condition="The two states are geographically adjacent.", 
                    df = shipment, 
                    depend_on = ['state_x', 'state_y'],
                    thinking = args.thinking)
    pred = df[['cust_name']].drop_duplicates()
    return f1_score(truth, pred), op.get_tokens()

# medium
# hybrid reasoning
def pipeline_select111(args):
    query = "Which Yelp restaurants with more than 3000 votes are located in a northeastern city of the United States?"
    truth = pd.DataFrame(['Joe’s Shanghai', 'The Halal Guys', 'Totto Ramen', 'Eataly NYC', 'Katz’s Delicatessen', 'Lombardi’s Pizza', 'Peter Luger Steak House', 'Ippudo NY'])
    
    yelp = pd.read_csv("./databases/restaurants2/yelp.csv")
    yelp = yelp[yelp['votes'] > 3000]
    print(yelp.shape[0]) # 22
    op = LogicalSelect(operand_type=OperandType.ROW)
    yelp = op.execute(impl_type=args.impl, 
                        condition="The zip belongs to a northeast city of USA", 
                        df = yelp, 
                        depend_on = ['zip'],
                        thinking = args.thinking)
    pred = yelp[['name']]
    return f1_score(truth, pred), op.get_tokens()

# easy
# semantic reasoning
def pipeline_select112(args):
    query = "Among the top 30 schools with the highest FRPM(K-12) counts, which ones have a human-sounding name?"
    truth  = pd.DataFrame(['John H. Francis Polytechnic', 'Hector G. Godinez', 'James A. Garfield Senior High'])
    
    schools = pd.read_csv("./databases/california_schools/schools.csv")
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")
    
    merged_df = pd.merge(schools, frpm, on='CDSCode')
    merged_df = merged_df.dropna(subset=['FRPM Count (K-12)'])
    merged_df = merged_df.sort_values(by='FRPM Count (K-12)', ascending=False)[:30]
    merged_df = merged_df[['County', 'School Name']]
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This schools sounds like a human's full name.",
                        df = merged_df, 
                        depend_on = ['School Name'],
                        thinking = args.thinking)
    pred = result[['School Name']]
    return f1_score(truth, pred), op.get_tokens()

# hard 
# semantic reasoning
def pipeline_select113(args):
    query = "List the order numbers of shipped orders whose comments request a specific delivery company."
    truth = pd.DataFrame([10109, 10215, 10254, 10308, 10313, 10319, 10336, 10340, 10358, 10400, 10413])
    orders = pd.read_csv("./databases/car_retails/orders.csv")
    orders = orders[orders['status'] == 'Shipped']
    orders = orders.dropna(subset=['comments'])
    # print(orders.shape[0]) # 62
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This comment appoints or requests specific company for delivering.",
                        df = orders, 
                        depend_on = ['comments'],
                        thinking=args.thinking)
    pred = result[['orderNumber']].drop_duplicates()
    return f1_score(truth, pred), op.get_tokens()


# easy
# fuzzy select
def pipeline_select114(args):
    query = "List the titles of NC-17 rated Action movies which are described as 'Reflection'."
    truth = pd.DataFrame(['DANCES NONE', 'DARKO DORADO', 'DRAGON SQUAD', 'MONTEZUMA COMMAND', 'REAR TRADING', 'UPRISING UPTOWN'])
    film = pd.read_csv("./databases/movie_3/film.csv")
    film = film[film['rating']=='NC-17']
    # print(film.shape[0]) # 194
    film_category = pd.read_csv("./databases/movie_3/film_category.csv")
    category = pd.read_csv("./databases/movie_3/category.csv")

    merged = pd.merge(film, film_category, on='film_id')
    merged = pd.merge(merged, category, on='category_id')
    merged = merged[merged['name'] == 'Action']
    # print(merged['description']) # 12

    op = LogicalSelect(operand_type=OperandType.ROW)
    result =op.execute(impl_type = args.impl, 
                        condition = "The movie is described as a Reflection",
                        df = merged,
                        depend_on = ['description'],
                        thinking = args.thinking)
    pred = result[['title']]
    return f1_score(truth, pred), op.get_tokens()

# medium
# world knowledge
def pipeline_select115(args):
    query = "List the seas whose depth is greater than the height of the highest peak of the Alps."
    truth = pd.DataFrame(['Arabian Sea', 'Arctic Ocean', 'Atlantic Ocean', 'Caribbean Sea', 'Gulf of Aden', 'Indian Ocean', 'Mediterranean Sea', 'Pacific Ocean', 'South China Sea', 'Sulawesi Sea', 'Sunda Sea'])
    sea = pd.read_csv("./databases/mondial_geo/sea.csv") # 37

    op = LogicalSelect(operand_type=OperandType.ROW)
    sea = op.execute(impl_type = args.impl, 
                    condition = "The depth is greater than the height of the highest peak of the Alps",
                    df = sea,
                    depend_on = ['Depth'],
                    thinking = args.thinking)
    pred = sea[['Name']]
    return f1_score(truth, pred), op.get_tokens()

# medium
# world knowledge
def pipeline_select116(args):
    query = "List the lakes with an altitude greater than 500 and an area larger than that of Shanghai."
    lake = pd.read_csv("./databases/mondial_geo/lake.csv")
    truth = pd.DataFrame(['Lake Bangweulu', 'Lake Tanganjika', 'Lake Titicaca', 'Lake Victoria', 'Salar de Uyuni'])
    lake = lake[lake['Altitude'] > 500]
    print(lake.shape) # 43
    op = LogicalSelect(operand_type=OperandType.ROW)
    lake = op.execute(impl_type = args.impl, 
                    condition = "The Area is larger than the area of Shanghai",
                    df = lake,
                    depend_on = ['Area'],
                    thinking = args.thinking)
    # print(lake['Name'].tolist())
    pred = lake[['Name']]
    return f1_score(truth, pred), op.get_tokens()

# easy
# world knowledge
def pipeline_select117(args):
    query = "List the IDs of customers located in the northeastern United States."
    truth = pd.DataFrame([600, 660, 3073, 4421, 3660, 4561])
    customer = pd.read_csv("./databases/shipping/customer.csv")
    print(customer.shape[0])
    op = LogicalSelect(operand_type=OperandType.ROW)
    customer = op.execute(impl_type=args.impl, 
                condition="The customer is located in the northeast side of USA", 
                df = customer, 
                depend_on = ['city', 'state'],
                thinking = args.thinking)
    pred = customer[['cust_id']]
    return f1_score(truth, pred), op.get_tokens()

# hard
# hybrid reasoning
def pipeline_select118(args):
    query = "Among the most recently updated schools, select the CDSCode for those within a 3-hour drive from San Francisco."
    truth = pd.DataFrame([44698490000000, 44698490126920, 44698496066542, 4615490000000])
    schools = pd.read_csv("./databases/california_schools/schools.csv")
    schools = schools.sort_values(by='LastUpdate', ascending=False)
    schools = schools[schools['LastUpdate'] == schools['LastUpdate'].tolist()[0]]
    # print(schools[['County', 'City', 'State']])
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This school is within a 3-hour drive from San Francisco.",
                        df = schools, 
                        depend_on = ['County', 'City', 'State'],
                        thinking = args.thinking)
    pred = result[['CDSCode']]
    return f1_score(truth, pred), op.get_tokens()

# medium
# numerical reasoning
def pipeline_select119(args):
    query = "Among the 50 most recent transactions paid in CZK, list the IDs of those with a price greater than 40 US dollars."
    truth = pd.DataFrame([988, 947, 978])
        
    transactions = pd.read_csv("./databases/debit_card_specializing/transactions_1k.csv")
    transactions['Time'] = transactions['Date'] + ' ' + transactions['Time']
    transactions = transactions.sort_values(by='Time', ascending=False)[:50]
    # print(transactions['Price'])
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "The price in CZK as currency is larger than 40 USD.",
                        df = transactions, 
                        depend_on = ['Price'],
                        thinking = args.thinking)
    pred = result[['TransactionID']]
    return f1_score(truth, pred), op.get_tokens()



def pipeline_select120(args):
    query = "List the ward numbers of Chicago wards whose alderman is a Democrat."
    truth = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50])
    ward = pd.read_csv("./databases/chicago_crime/Ward.csv")
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This Chicago ward is a Democrat.",
                        df = ward, 
                        depend_on = ['alderman_first_name','alderman_last_name'],
                        thinking=args.thinking)
    pred = result[['ward_no']]
    return f1_score(truth, pred), op.get_tokens()


def pipeline_select121(args):
    query = "List the code numbers of FBI agents who are responsible for investigating sexual crimes."
    truth = pd.DataFrame(['2', '16', '17'])
    
    fbi = pd.read_csv("./databases/chicago_crime/FBI_Code.csv")
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This FBI agent is responsible for investigating sexual crimes.",
                        df = fbi, 
                        depend_on = ['description'],
                        thinking=args.thinking)
    pred = result[['fbi_code_no']]
    print(pred)
    return f1_score(truth, pred), op.get_tokens()
    
def pipeline_select122(args):
    query = "Among the 50 most recently created users with a non-empty AboutMe, list the IDs of those whose AboutMe contains an external link."
    truth = pd.DataFrame([55675, 55645, 55642, 55635, 55578, 55566, 55558])
    
    users = pd.read_csv("./databases/codebase_community/users.csv")
    users = users.dropna(subset=['AboutMe'])
    users = users.sort_values(by='CreationDate', ascending=False)[:50]
        
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition = "This personal introudction contains external link.",
                        df = users, 
                        depend_on = ['AboutMe'],
                        thinking=args.thinking)
    pred = result[['Id']]
    return f1_score(truth, pred), op.get_tokens()


# external knowledge, medium
def pipeline_select123(args):
    query = "List all languages that originate from an Asian country."
    truth = pd.DataFrame(['Japanese', 'Arabic', 'Chinese', 'Malaysian', 'Turkish'])
    book_language = pd.read_csv("./databases/books/book_language.csv")
    op = LogicalSelect(operand_type=OperandType.ROW)
    book_language = op.execute(impl_type=args.impl,
                        condition = "This language originates from an Asian country.",
                        df = book_language, 
                        depend_on = ['language_name'],
                        thinking = args.thinking)
    pred = book_language[['language_name']]
    return f1_score(truth, pred), op.get_tokens()

# fuzzy, easy
def pipeline_select124(args):
    query = "List all the languages that are English or a variation of English."
    truth = pd.DataFrame(['English', 'United States English', 'British English', 'Middle English', 'Canadian English'])
    book_language = pd.read_csv("./databases/books/book_language.csv")
    op = LogicalSelect(operand_type=OperandType.ROW)
    book_language = op.execute(impl_type=args.impl,
                        condition = "This language is English or its variation.",
                        df = book_language, 
                        depend_on = ['language_name'],
                        thinking = args.thinking)
    pred = book_language[['language_name']]
    return f1_score(truth, pred), op.get_tokens()


def pipeline_select125(args):
    query = "Among the 50 least recently added animals, list the breeds of those with no less than three fur colors."
    truth = pd.DataFrame(['Terrier', 'Cat - DMH Tortie/Tabby', 'Cat - DLH Tabby', 'Shepherd X', 'Eng. Bulldog', 'Cat - DSH Abyssinian', 'Cat - DSH Tabby'])
    animals = pd.read_csv("./databases/nextiaJD/animal-control-inventory-lost-and-found.csv")
    animals = animals.sort_values(by='DateCreated', ascending=False)[:50]
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl, 
                        condition="The animal has three colors of fur",
                        df = animals,
                        depend_on = 'Color', 
                        thinking = args.thinking)
    pred = result[['Breed']]
    return f1_score(truth, pred), op.get_tokens()

# external knowledge, easy
def pipeline_select126(args):
    query = "Among the 50 least recently updated movies, list the titles of those with a longer runtime than 'The Silence of the Lambs'."
    truth = pd.DataFrame(['PAST SUICIDES', 'PATTON INTERVIEW', 'PAYCHECK WAIT', 'PEACH INNOCENT', 
                          'PHILADELPHIA WIFE', 'PIANIST OUTFIELD', 'PITTSBURGH HUNCHBACK', 'PIZZA JUMANJI', 
                          'PLATOON INSTINCT', 'PARIS WEEKEND', 'PAPI NECKLACE', 'NORTHWEST POLISH', 
                          'NOTORIOUS REUNION', 'NUTS TIES', 'OLEANDER CLUE', 'OPEN AFRICAN', 
                          'OPERATION OPERATION', 'ORDER BETRAYED'])
    film = pd.read_csv("./databases/movie_3/film.csv")
    film = film.sort_values(by='last_update', ascending=False)[:50]
    # print(film.shape[0]) # 194
    op = LogicalSelect(operand_type=OperandType.ROW)
    film = op.execute(impl_type = args.impl, 
                        condition = "The movie has a longer runtime than 'The Silence of the Lambs'",
                        df = film,
                        depend_on = ['length'],
                        thinking = args.thinking)
    pred = film[['title']]
    return f1_score(truth, pred), op.get_tokens()

def pipeline_select127(args):
    query = "Among players born in 1998, select ids of those whose height is taller than Barack Obama's."
    truth = pd.DataFrame([4143, 4754, 4819, 5917, 8840, 10334])
    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player = player[pd.to_datetime(player['birthday']).dt.year == 1998]
    op = LogicalSelect(OperandType.ROW)
    player = op.execute(impl_type=args.impl, 
                        condition="The player's height is taller than Barack Obama's height", 
                        df = player, 
                        depend_on = 'height',
                        thinking=args.thinking)
    pred = player[['id']]
    print(player['id'].tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_select128(args):
    query = "Filter Disney characters where the movie title contains the hero's name, and output the movie titles."
    truth = pd.DataFrame(['Snow White and the Seven Dwarfs', 'Pinocchio', 'Dumbo', 'Bambi', 
                          'The Adventures of Ichabod and Mr. Toad', 'Cinderella', 'Alice in Wonderland', 
                          'Peter Pan', 'Lady and the Tramp', 'Robin Hood', 'The Many Adventures of Winnie the Pooh', 
                          'Oliver & Company', 'Aladdin', 'Pocahontas', 'Hercules', 'Mulan', 'Tarzan', 'Lilo & Stitch', 
                          'Bolt', 'Winnie the Pooh', 'Wreck-It Ralph', 'Big Hero 6', 'Moana'])
    characters = pd.read_csv("./databases/disney/characters.csv")
    op = LogicalSelect(OperandType.ROW)
    characters = op.execute(impl_type=args.impl, 
                            condition = "The movie title contains the hero name in this movie",
                            df = characters, 
                            depend_on = ['movie_title', 'hero'], 
                            thinking=args.thinking)
    pred = characters[['movie_title']]
    # print(characters['movie_title'].tolist())
    return f1_score(truth, pred), op.get_tokens()


def pipeline_select129(args):
    query = "Filter Disney movies where the country in which the movie was filmed is different from the director's country of birth, and output the movie titles."
    truth = pd.DataFrame(['Saludos Amigos', 'Melody Time', 'Alice in Wonderland', 'Sleeping Beauty', '101 Dalmatians', 
                          'The Sword in the Stone', 'The Jungle Book', 'The Aristocats', 'Robin Hood', 
                          'The Many Adventures of Winnie the Pooh', 'The Rescuers', 'Oliver & Company', 
                          'The Rescuers Down Under', 'Aladdin', 'Mulan', 'Dinosaur', 'Brother Bear'])
    director = pd.read_csv("./databases/disney/director.csv")
    op = LogicalSelect(OperandType.ROW)
    director = op.execute(impl_type=args.impl, 
                            condition = "The country where the movie was filmed is different from the director's country of birth.",
                            df = director, 
                            depend_on = ['name', 'director'], 
                            thinking=args.thinking)
    print(director['name'].tolist())
    pred = director[['name']]
    return f1_score(truth, pred), op.get_tokens()


# Column-Level
# Query Relevance Selection
def pipeline_select200(args):
    query = "Select all the columns related to the SAT test in the 'satscores' table."
    truth = ['NumTstTakr', 'AvgScrRead', 'AvgScrWrite', 'AvgScrMath','NumGE1500']
    scores = pd.read_csv("./databases/california_schools/satscores.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type = args.impl,
                    condition = "The column is related to the SAT test.",
                    df = scores, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Value Pattern Selection
def pipeline_select201(args):
    query = "Which team attribute columns can be used for performance classification?"
    truth = [
        'buildUpPlaySpeedClass','buildUpPlayDribblingClass','buildUpPlayPassingClass', 'buildUpPlayPositioningClass', 'chanceCreationPassingClass',
        'chanceCreationCrossingClass','chanceCreationShootingClass','chanceCreationPositioningClass','defencePressureClass','defenceAggressionClass',
        'defenceTeamWidthClass','defenceDefenderLineClass'
    ]
    team_attribute = pd.read_csv("./databases/european_football_2/Team_Attributes.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column has a predefined string tag sets for team performance classification.",
                    df = team_attribute, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Query Relevance Selection
def pipeline_select202(args):
    query = "When querying for a superhero's name or appearance, which columns are relevant in the 'superhero' table?"
    truth = ['superhero_name', 'full_name', 'eye_colour_id', 'hair_colour_id', 'skin_colour_id']
    superhero = pd.read_csv("./databases/superhero/superhero.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to superhero's name or apperance.",
                    df = superhero, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Query Relevance Selection
def pipeline_select203(args):
    query = "Which columns are related to money in the 'transactions_1k' table?"
    truth = ['Amount', 'Price']
    transactions_1k = pd.read_csv("./databases/debit_card_specializing/transactions_1k.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to money.",
                    df = transactions_1k, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Value Pattern Selection
def pipeline_select204(args):
    query = "Which columns are relevant when querying for date attributes in the 'posts' table?"
    truth = ['CreaionDate', 'LasActivityDate', 'LastEditDate','CommunityOwnedDate', 'ClosedDate']
    posts = pd.read_csv("./databases/codebase_community/posts.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column might have a date field.",
                    df = posts, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Value Pattern Selection
def pipeline_select205(args):
    query = "List all columns that represent temporal references in the 'amazon_music' table."
    truth = ['Time', 'Released', 'Copyright']
    music = pd.read_csv("./databases/music/amazon_music.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column might be a temporal reference.",
                    df = music, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Query Relevance Selection
def pipeline_select206(args):
    query = "List the columns in the 'city' table that are relevant to administrative division attributes or concepts."
    truth = ['Name', 'Country', 'Province', 'Population']
    cities = pd.read_csv("./databases/mondial_geo/city.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to a administrative division attribute or concept.",
                    df = cities, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Value Pattern Selection
def pipeline_select207(args):
    query = "Which columns are related to device configuration in the 'amazon' table?"
    truth = ['Name', 'Features']
    products = pd.read_csv("./databases/electronics/amazon.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to device configuration.",
                    df = products, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Query Relevance Selection
def pipeline_select208(args):
    query = "Which columns are related to contact information in the 'ward' table?"
    truth = ['ward_office_address', 'ward_office_zip', 'ward_email', 'ward_office_phone', 'ward_office_fax', 'city_hall_office_phone', 'city_hall_office_fax']
    ward = pd.read_csv("./databases/chicago_crime/Ward.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to contact information.",
                    df = ward, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

# Query Relevance Selection
def pipeline_select209(args):
    query = "List the columns related to restaurant popularity in the 'zomato' table."
    truth = ['votes', 'rating', 'reviewcount']
    rests = pd.read_csv("./databases/restaurants2/zomato.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to restaurant popularity.",
                    df = rests, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

def pipeline_select210(args):
    query = "List all columns related to player names in the 'Master' table."
    truth = ['firstName', 'lastName', 'nameNote', 'nameGiven', 'nameNick']
    master = pd.read_csv("./databases/hockey/Master.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to the player name.",
                    df = master, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

def pipeline_select211(args):
    query = "List all columns related to shooting in the 'Scoring' table."
    truth = ['G', 'PPG', 'SHG', 'GWG', 'GTG', 'SOG', 'PostG', 'PostPPG', 'PostSHG', 'PostGWG', 'PostSOG']
    scoring = pd.read_csv("./databases/hockey/Scoring.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to shooting.",
                    df = scoring, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

def pipeline_select212(args):
    query = "List all columns related to financial value in the 'Contracts-2019-2020-q1' table."
    truth = ['Estimated value', 'Estimated annual value', 'VAT not recovered']
    contract = pd.read_csv("./databases/santos/Contracts-2019-2020-q1.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to financial value.",
                    df = contract, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

def pipeline_select213(args):
    query = "List all columns related to contract terms in the 'Contracts-2019-2020-q1' table."
    truth = ['Initial contract period (months)', 'Total option to extend (months)', 'Total contract period (months)', 
    'Available No. of Ext.', 'Available extension details', 'Taken No. of Ext.', 'Taken Extension details']
    contract = pd.read_csv("./databases/santos/Contracts-2019-2020-q1.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column is related to contract term.",
                    df = contract, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()

def pipeline_select214(args):
    query = "Which columns should I refer to in order to understand the business activity of locations in the 'Copy2520of2520Monthly2520Data2520Feed-June25202016' table?"
    truth = ['BRCYear', 'BRCQuarter', 'BRCMonthName', 'BRCWeek','BusinessInCount', 'BusinessOutCount', 'BusinessTotalCount']
    contract = pd.read_csv("./databases/santos/Copy2520of2520Monthly2520Data2520Feed-June25202016.csv")

    op = LogicalSelect(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl,
                    condition = "The column relates to business.",
                    df = contract, 
                    example_num = args.example_num,
                    thinking = args.thinking)
    return f1_score(truth, result.columns.tolist()), op.get_tokens()


# Table-Level 
# Semantic Table Selection
def pipeline_select300(args):
    query = "List all tables related to driver information."
    truth = ['driverStandings', 'drivers', 'lapTimes', 'pitStops', 'qualifying', 'results']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/formula_1"):
        csv_path = os.path.join("./databases/formula_1", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    
    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="The table contains some information related to the drivers.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    print(pred)
    return f1_score(truth, pred), op.get_tokens()
    
    
# Semantic Table Selection
# interesting case (LLM_SEMI accuracy is zero, LLM_ONLY accuracy is 100%.)
def pipeline_select301(args):
    query = "List all tables related to book attributes and information."
    truth = ['book', 'book_author', 'author', 'book_language', 'publisher']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/books"):
        csv_path = os.path.join("./databases/books", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="The table is related to book attributeds and information.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

# Semantic Table Selection
def pipeline_select302(args):
    query = "List all tables related to geographic locations where crimes occur."
    truth = ['Community_Area', 'Crime', 'District', 'Neighborhood', 'Ward']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/chicago_crime"):
        csv_path = os.path.join("./databases/chicago_crime", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="The table is related to a geographic location where crimes occur.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

# Semantic Table Selection
def pipeline_select303(args):
    query = "List all tables related to comments."
    truth = ['comments', 'posts', 'users']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/codebase_community"):
        csv_path = os.path.join("./databases/codebase_community", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="The table is related to comments.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

# Semantic Table Selection
def pipeline_select304(args):
    query = "List all tables related to a superhero's appearance."
    truth = ['colour', 'gender', 'race', 'superhero']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/superhero"):
        csv_path = os.path.join("./databases/superhero", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="The table is related to superhero's apperance.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

# Semantic Table Selection
def pipeline_select305(args):
    query = "List all tables related to football leagues."
    truth = ['Country', 'League', 'Match']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/european_football_2"):
        csv_path = os.path.join("./databases/european_football_2", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="The table is related to football league.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

# Knowledge Table Selection
def pipeline_select306(args):
    query = "Which tables are needed to retrieve Lewis Hamilton's average lap time in the 2016 Singapore Grand Prix?"
    truth = ['drivers','lapTimes','races']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/formula_1"):
        csv_path = os.path.join("./databases/formula_1", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="From this table Lewis Hamilton's average laptime in 2016 Signapore Grand Prix can be retrieved.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

# Knowledge Table Selection
def pipeline_select307(args):
    query = "Which tables are needed to retrieve the matches that FC Barcelona participated in?"
    truth = ['Match', 'Team']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/european_football_2"):
        csv_path = os.path.join("./databases/european_football_2", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="From this table matches FC Barcelona participated in can be retrieved.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

# Semantic Table Selection
def pipeline_select308(args):
    query = "List the tables relevant to geographic concepts."
    truth = ['address', 'city', 'country']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/movie_3"):
        csv_path = os.path.join("./databases/movie_3", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="This table is related to geographic concepts.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

# Semantic Table Selection
def pipeline_select309(args):
    query = "Select all tables that are not related to terrain."
    truth = [
        'borders', 'city', 'continent', 'country', 'economy', 'encompasses', 
        'ethnicGroup', 'isMember', 'language', 'organization', 'politics', 
        'population', 'province', 'religion', 'target'
    ]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/mondial_geo"):
        csv_path = os.path.join("./databases/mondial_geo", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="This table is not realated to terrain.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()


def pipeline_select310(args):
    query = "Select all tables related to the water system."
    truth = ['geo_estuary', 'geo_lake', 'geo_river', 'geo_sea', 'geo_source', 'lake', 'located', 'river', 'sea', 'islandIn', 'mergesWith']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/mondial_geo"):
        csv_path = os.path.join("./databases/mondial_geo", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="This table relates to the water system.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()

def pipeline_select311(args):
    query = "Select all tables containing information about coaches."
    truth = ['Coaches', 'Master', 'AwardsCoaches']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/hockey"):
        csv_path = os.path.join("./databases/hockey", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="This table contains information about coaches.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    print(pred)
    return f1_score(truth, pred), op.get_tokens()

def pipeline_select312(args):
    query = "If I want to know the revenue generated by a specific author, which tables should I refer to?"
    truth = ['author', 'book_author', 'order_line']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/books"):
        csv_path = os.path.join("./databases/books", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="This table satisfies 'If I want to know the revenue of a specific author, I need to refer to this table'.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    print(pred)
    return f1_score(truth, pred), op.get_tokens()

def pipeline_select313(args):
    query = "If I want to identify the customer who spends the most money buying books, which tables should I refer to?"
    truth = ['cust_order', 'order_line', 'customer']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/books"):
        csv_path = os.path.join("./databases/books", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="If I want to know the customer who spends the most money buying books, I should refer to this table.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    print(pred)
    return f1_score(truth, pred), op.get_tokens()


def pipeline_select314(args):
    query = "If I want to find all players who have won awards both as a goalie and as a coach, which tables should I refer to?"
    truth = ['Coaches', 'AwardsPlayers', 'AwardsCoaches', 'Master']
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/hockey"):
        csv_path = os.path.join("./databases/hockey", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))

    op = LogicalSelect(operand_type=OperandType.TABLE)
    pred = op.execute(impl_type=args.impl, 
                                condition="If I want to know all the players who have been and won awards as a goalie and a coach, I need to refer to this table.", 
                                df = table_dfs, 
                                table_names = table_names, 
                                example_num = args.example_num,
                                thinking = args.thinking)
    print(pred)
    return f1_score(truth, pred), op.get_tokens()

if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(impl=ImplType.LLM_ALL, thinking=False, example_num=3)
    print(pipeline_select110(args))
