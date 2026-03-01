import pandas as pd 
import os 
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import LogicalMatch
from src.core.enums import OperandType, ImplType
from src.metrics.classic import f1_score

# fuzzy join, medium
def pipeline_match100(args):
    query = "Match 2009 races to circuits with latitude no less than 50, under the condition that the race is held in the circuit country."
    truth = pd.DataFrame([
        (8,  9),
        (8, 31),
        (8, 38),
        (8, 58),
        (9, 20),
        (9, 61),
        (12, 13),
        (12, 40),
        (12, 50),
    ])
    races = pd.read_csv("./databases/formula_1/races.csv")  
    races = races[races['year'] == 2009] 
    circuits = pd.read_csv("./databases/formula_1/circuits.csv") 
    circuits = circuits[circuits['lat'] >= 50] 
    # print(races.shape[0], circuits.shape[0]) 
    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The race 'name' is held in the 'country'", 
                            left_df=races, 
                            right_df=circuits, 
                            left_on='name', 
                            right_on='country',
                            thinking = args.thinking) 
    pred = merged_df[['left_raceId', 'right_circuitId']]
    return f1_score(truth, pred), op.get_tokens()

# fuzzy join, easy
def pipeline_match101(args):
    query = "Match the last 10 Disney characters' release dates to the corresponding revenue years."
    truth = pd.DataFrame([
        ('30-Mar-07', 2007),
        ('21-Nov-08', 2008),
        ('11-Dec-09', 2009),
        ('24-Nov-10', 2010),
        ('15-Jul-11', 2011),
        ('2-Nov-12', 2012),
        ('27-Nov-13', 2013),
        ('7-Nov-14', 2014),
        ('4-Mar-16', 2016),
        ('23-Nov-16', 2016),
    ])
    characters = pd.read_csv("./databases/disney/characters.csv")
    revenue = pd.read_csv("./databases/disney/revenue.csv")
    characters = characters[-10:]

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The 'release_date' corresponds to the 'Year'", 
                            left_df=characters, 
                            right_df=revenue, 
                            left_on='release_date', 
                            right_on='Year',
                            thinking = args.thinking) 
    pred = merged_df[['left_release_date', 'right_Year']]
    return f1_score(truth, pred), op.get_tokens()
    
# semantic join, hard
def pipeline_match102(args):
    query = "Match customers with a credit limit greater than 100,000 to the offices' territories where their cities are located."
    truth = pd.DataFrame([
        (114, 'APAC'),
        (119, 'EMEA'),
        (141, 'EMEA'),
        (146, 'EMEA'),
        (148, 'APAC'),
        (157, 'NA'),
        (187, 'EMEA'),
        (227, 'EMEA'),
        (249, 'EMEA'),
        (259, 'EMEA'),
        (276, 'APAC'),
        (278, 'EMEA'), 
        (286, 'NA'),
        (298, 'EMEA'), 
        (319, 'NA'),
        (363, 'NA'),
        (386, 'EMEA'),
        (448, 'EMEA'),
        (458, 'EMEA'),
        (496, 'APAC'),
    ])
    customers = pd.read_csv("./databases/car_retails/customers.csv")
    customers = customers[customers['creditLimit'] > 100000]
    # print(customers)
    offices = pd.read_csv("./databases/car_retails/offices.csv")[['territory']].drop_duplicates()

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The 'left_city' locates in the 'right_territory'.", 
                            left_df=customers, 
                            right_df=offices, 
                            left_on='city', 
                            right_on='territory',
                            thinking = args.thinking) 
    # print(merged_df[['left_customerNumber', 'left_city', 'right_territory']])
    pred = merged_df[['left_customerNumber', 'right_territory']]
    return f1_score(truth, pred), op.get_tokens()

# semantic reasoning, hard
def pipeline_match103(args):
    query = "Match films whose titles start with 'U' to the categories based on their descriptions."
    film = pd.read_csv("./databases/movie_3/film.csv")
    film = film[film['title'].str.startswith('U')][['film_id', 'title', 'description']]
    # print(film.shape[0])
    category = pd.read_csv("./databases/movie_3/category.csv")
    # print(category.shape[0])

    mid = pd.read_csv("./databases/movie_3/film_category.csv")
    mid = pd.merge(film, mid, on='film_id')
    mid = pd.merge(mid, category, on='category_id')
    truth = mid[['film_id', 'category_id']]

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The 'description' of the movie belongs to the 'name' category ", 
                            left_df=film, 
                            right_df=category, 
                            left_on='description', 
                            right_on='name',
                            thinking = args.thinking) 
    pred = merged_df[['left_film_id', 'right_category_id']]
    # print(truth)
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()

# semantic join, medium
def pipeline_match104(args):
    query = "Match the first 10 comments to the top 10 most popular tags based on the relevance of the comment content to the tag name."
    truth = pd.DataFrame([
        (2,  41),
        (4, 9),
        (9, 41),
        (11, 41)
    ])
    comments = pd.read_csv("./databases/codebase_community/comments.csv")
    comments = comments[:10]
    comments = comments[['Id', 'Text', 'Score']]
    tags =  pd.read_csv("./databases/codebase_community/tags.csv")
    tags = tags.sort_values(by='Count', ascending=False)[:10]

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The content of the comment 'Text' is related to the 'TagName'", 
                            left_df=comments, 
                            right_df=tags, 
                            left_on='Text', 
                            right_on='TagName',
                            thinking = args.thinking) 
    pred = merged_df[['left_Id', 'right_Id']]
    return f1_score(truth, pred), op.get_tokens()

# semantic join, hard
def pipeline_match105(args):
    query = "Match wholesaler customers to states with cities having a population greater than 600,000, by matching their abbreviated state codes to the full state names."
    truth = pd.DataFrame([
        (381, 'Florida'),
        (615, 'Florida'),
        (618, 'Florida'),
        (1575, 'California'),
        (1685, 'California'),
        (1769, 'Texas'),
        (2001, 'California'),
        (2799, 'Texas'),
        (3288, 'Texas'),
        (3660, 'Maryland'),
        (4353, 'Texas'),
        (4869, 'Arizona'),
    ])
    customer = pd.read_csv("./databases/shipping/customer.csv")
    customer = customer[customer['cust_type']=='wholesaler']
    customer = customer[['cust_id', 'cust_name', 'state']]
    city = pd.read_csv("./databases/shipping/city.csv")
    city = city[city['population']>600000]
    city = city[['state']].drop_duplicates()
    # print(customer)
    # print(city)
    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The abbreviated 'left_state' is exactly the 'right_state'", 
                            left_df=customer, 
                            right_df=city, 
                            left_on='state', 
                            right_on='state',
                            thinking = args.thinking) 
    pred = merged_df[['left_cust_id', 'right_state']]
    # print(merged_df[['left_cust_id', 'left_state', 'right_state']])
    return f1_score(truth, pred), op.get_tokens()

# semantic join, hard
def pipeline_match106(args):
    query = "Match customers with annual revenue greater than 45,000,000 to states with cities having a population greater than 600,000, by matching their abbreviated state codes to the full state names."
    truth = pd.DataFrame([
        (721, 'Florida'),
        (1724, 'Ohio'),
        (2001, 'California'),
        (2434, 'Ohio'),
        (2799, 'Texas'),
        (3320, 'California'),
        (3447, 'California'),
    ])
    customer = pd.read_csv("./databases/shipping/customer.csv")
    customer = customer[customer['annual_revenue']>45000000]
    customer = customer[['cust_id', 'cust_name', 'state']]
    city = pd.read_csv("./databases/shipping/city.csv")
    city = city[city['population']>600000]
    city = city[['state']].drop_duplicates()
    # print(customer)
    # print(city)
    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The abbreviated 'left_state' is exactly the 'right_state'", 
                            left_df=customer, 
                            right_df=city, 
                            left_on='state', 
                            right_on='state',
                            thinking = args.thinking) 
    pred = merged_df[['left_cust_id', 'right_state']]
    # print(merged_df[['left_cust_id', 'left_state', 'right_state']])
    return f1_score(truth, pred), op.get_tokens()
    

# semantic join, easy 
def pipeline_match107(args):
    query = "Match football teams whose names start with 'P' to the countries they belong to."
    truth = pd.DataFrame([
        (3476, 1729),
        (9548, 4769),
        (20532, 10257),
        (21292, 10257),
        (23523, 10257),
        (26556, 13274),
        (29000, 13274),
        (31444, 15722),
        (31445, 15722),
        (31448, 15722),
        (31457, 15722),
        (32891, 15722),
        (33377, 15722),
        (36248, 17642),
        (41673, 19694),
    ])
    team = pd.read_csv("./databases/european_football_2/Team.csv")
    team = team[team['team_long_name'].str.startswith('P')]
    team = team[['id', 'team_long_name']]
    # print(team)
    # print(team.shape[0])
    country = pd.read_csv("./databases/european_football_2/Country.csv")
    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The 'left_team_long_name' team belongs to the 'right_name' country", 
                            left_df=team, 
                            right_df=country, 
                            left_on='team_long_name', 
                            right_on='name',
                            thinking = args.thinking)
    pred = merged_df[['left_id', 'right_id']]
    return f1_score(truth, pred), op.get_tokens()

# external info join, hard
def pipeline_match108(args):
    query = "Match superheroes with a height greater than 300 cm to their publishers."
    superhero = pd.read_csv("./databases/superhero/superhero.csv")
    superhero = superhero[superhero['height_cm']>300]
    publisher = pd.read_csv("./databases/superhero/publisher.csv").dropna()
    truth = pd.merge(superhero, publisher, left_on='publisher_id', right_on='id')
    truth = truth[['superhero_name', 'publisher_name']]
    

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The superhero in 'left_superhero_name' is created by the publisher in 'right_publisher_name'", 
                            left_df=superhero, 
                            right_df=publisher, 
                            left_on='superhero_name', 
                            right_on='publisher_name',
                            thinking = args.thinking)
    pred = merged_df[['left_superhero_name', 'right_publisher_name']]
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# external info join, hard
def pipeline_match109(args):
    query = "Match customers with annual revenue greater than 45,000,000 to states with cities having a population greater than 600,000, where the customer's state is adjacent to the city's state."
    truth = pd.DataFrame([[314, 'Illinois'], [931, 'California'], [954, 'Texas'], 
                          [954, 'Arizona'], [1290, 'Arizona'], [1592, 'Texas'], 
                          [1592, 'Tennessee'], [1669, 'Maryland'], [1669, 'Ohio'], 
                          [1669, 'Illinois'], [1669, 'Tennessee'], [1669, 'Indiana'], 
                          [1724, 'Pennsylvania'], [1724, 'Michigan'], [1724, 'Indiana'], 
                          [2001, 'Arizona'], [2421, 'Michigan'], [2421, 'Illinois'], 
                          [2434, 'Pennsylvania'], [2434, 'Michigan'], [2434, 'Indiana'], 
                          [2496, 'California'], [3320, 'Arizona'], [3447, 'Arizona'], 
                          [4415, 'California'], [4426, 'Texas'], [4426, 'Arizona']])
    customer = pd.read_csv("./databases/shipping/customer.csv")
    customer = customer[customer['annual_revenue']>45000000]
    customer = customer[['cust_id', 'cust_name', 'state']]
    city = pd.read_csv("./databases/shipping/city.csv")
    city = city[city['population']>600000]
    city = city[['state']].drop_duplicates()
    # print(customer)
    # print(city)
    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl,
                            condition="The 'left_state' is adjacent to the 'right_state'", 
                            left_df=customer, 
                            right_df=city, 
                            left_on='state', 
                            right_on='state',
                            thinking = args.thinking) 
    pred = merged_df[['left_cust_id', 'right_state']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match110(args):
    query = "Match state expenditure agencies with payments total greater than 1,500 to Finance-related departments in the rm-mr-2009-eng dataset that share the same function."
    truth = pd.DataFrame([['COMMERCE AND INSURANCE', 'Financial Consumer Agency of Canada'], 
                          ['COMMERCE AND INSURANCE', 'Office of the Superintendent of Financial Institutions'], 
                          ['REVENUE', 'Finance Department'], 
                          ['OFFICE OF STATE AUDITOR', 'Auditor General'],
                          ['OFFICE OF STATE TREASURER','Finance Department']])
    state_expenditures = pd.read_csv("./databases/santos/state_expenditures.csv")
    rm_2009 = pd.read_csv("./databases/santos/rm-mr-2009-eng.csv")

    state_expenditures = state_expenditures[state_expenditures['Payments Total'] > 1500]
    rm_2009 = rm_2009[rm_2009['MINE']=='Finance']
    state_expenditures = state_expenditures[['Agency Name']]
    rm_2009 = rm_2009[['DEPT_EN_DESC']]
    state_expenditures = state_expenditures.drop_duplicates()

    op = LogicalMatch(OperandType.CELL)
    pred = op.execute(impl_type=args.impl, 
                           condition="The agency and the DEPT have the same function.", 
                           left_df = state_expenditures, 
                           right_df = rm_2009, 
                           left_on = 'Agency Name',
                           right_on = 'DEPT_EN_DESC',
                           thinking=args.thinking)
    # print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match111(args):
    query = "Match state expenditure agencies with payments total greater than 1,800 to Natural Resources departments in the rm-mr-2009-eng dataset that share the same function."
    truth = pd.DataFrame([['NATURAL RESOURCES', 'Natural Resources Department']])
    state_expenditures = pd.read_csv("./databases/santos/state_expenditures.csv")
    rm_2009 = pd.read_csv("./databases/santos/rm-mr-2009-eng.csv")

    state_expenditures = state_expenditures[state_expenditures['Payments Total'] > 1800]
    rm_2009 = rm_2009[rm_2009['MINE']=='Natural Resources']
    state_expenditures = state_expenditures[['Agency Name']]
    rm_2009 = rm_2009[['DEPT_EN_DESC']]
    state_expenditures = state_expenditures.drop_duplicates()
    print(state_expenditures)
    print(rm_2009)

    op = LogicalMatch(OperandType.CELL)
    pred = op.execute(impl_type=args.impl, 
                           condition="The agency and the DEPT have the same function.", 
                           left_df = state_expenditures, 
                           right_df = rm_2009, 
                           left_on = 'Agency Name',
                           right_on = 'DEPT_EN_DESC',
                           thinking=args.thinking)
    # print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match112(args):
    query = "Match transactions from the British Transport Police on 28/05/2010 in the DfT monthly spend dataset to Software coding transactions in the transparency report that share the same day of the month."
    truth = pd.DataFrame([['208445', 5100014873], ['208445', 5100014873], ['208445', 5100014873], ['208445', 5100014873], 
                          ['208445', 5100014873], ['212133', 5100014873], ['212133', 5100014873], ['212076', 5100014873], 
                          ['212077', 5100014873], ['208155', 5100014873], ['208155', 5100014873], ['208079', 5100014873], 
                          ['212164', 5100014873], ['211062', 5100014873], ['211062', 5100014873], ['211062', 5100014873], 
                          ['212355', 5100014873], ['210874', 5100014873], ['212283', 5100014873], ['211598', 5100014873], 
                          ['211598', 5100014873], ['212340', 5100014873], ['212340', 5100014873]])
    dft_monthly_spend = pd.read_csv("./databases/santos/dft-monthly-spend-201005.csv")
    transprncy_rpt = pd.read_csv("./databases/santos/V3.1_-_Final__csv__May_2013__25k_Transprncy_rpt.csv")
    dft_monthly_spend = dft_monthly_spend[dft_monthly_spend['Entity'] == 'British Transport Police']
    dft_monthly_spend = dft_monthly_spend[dft_monthly_spend['Date'] == '28/05/2010']
    dft_monthly_spend =dft_monthly_spend[['Date', 'Transaction number']]
    transprncy_rpt = transprncy_rpt[transprncy_rpt['Expense type'] == 'Software coding']
    transprncy_rpt = transprncy_rpt[['Date', 'Transaction number']]
    print(dft_monthly_spend.shape)
    print(transprncy_rpt.shape)
    
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The two date shares the same day of different time.", 
                           left_df = dft_monthly_spend, 
                           right_df = transprncy_rpt, 
                           left_on = 'Date',
                           right_on = 'Date',
                           thinking=args.thinking)
    print(merged_df)
    print(op.get_tokens())
    # print(merged_df[['left_Transaction number', 'right_Transaction number']].values.tolist())
    pred = merged_df[['left_Transaction number', 'right_Transaction number']]
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match113(args):
    query = "List the senior officials from the Home Office travel data return who share the same first name with officials in the April-June 2018 travel expenses report."
    truth = pd.DataFrame([['Mark Sedwill ', 'Mark Bryson-Richardson'], ['Richard Clarke', 'Richard Montgomery']])

    home_office_travel = pd.read_csv("./databases/santos/home_office_senior_officials_travel_data_return.csv")
    travel_exp_2018 = pd.read_csv("./databases/santos/travel-exp-April-June-2018.csv")
    
    # Filter to get unique official names from both datasets
    home_office_officials = home_office_travel[['Name of Official']].drop_duplicates()
    travel_2018_officials = travel_exp_2018[['Senior Officials Name', 'Destination']].drop_duplicates(subset=['Senior Officials Name'])

    print(len(home_office_officials), len(travel_2018_officials))
    
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The two senior government official names shares the same first name.", 
                           left_df=home_office_officials, 
                           right_df=travel_2018_officials, 
                           left_on='Name of Official',
                           right_on='Senior Officials Name',
                           thinking=args.impl)
    pred = merged_df[['left_Name of Official', 'right_Senior Officials Name']]
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match114(args):
    query = "Match pubs to community centres in the Barking and Dagenham borough that are located on the same street."
    truth = pd.DataFrame([['109 Bastable Avenue / Charlton Crescent', 'Bastable Avenue'], 
                          ['109 Bastable Avenue / Charlton Crescent', 'Bastable Avenue'], 
                          ['Billet Road / Rose Lane', 'Rose Lane']])
    pubs = pd.read_csv("./databases/santos/pubs.csv")
    community = pd.read_csv("./databases/santos/community_centres.csv")
    
    pubs = pubs[pubs['borough_name'] == 'Barking and Dagenham']
    community = community[community['borough_name'] == 'Barking and Dagenham']
    print(len(pubs), len(community))

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The two addresses represent the same street location.", 
                           left_df=pubs, 
                           right_df=community, 
                           left_on='address1',
                           right_on='address1',
                           thinking=args.thinking)
    pred = merged_df[['left_address1', 'right_address1']]
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match115(args):
    query = "Match each desert to the continent where its main part is located."
    truth = pd.DataFrame([['Arabian Desert', 'Asia'], ['Atacama', 'America'], ['Azaouad', 'Africa'], ['Baja California Desert', 'America'], ['Chihuahua', 'America'], ['Colorado Plateau', 'America'], ['Darfur', 'Africa'], 
                          ['Dascht-e-Kavir', 'Asia'], ['Dascht-e-Lut', 'Asia'], ['Dascht-e-Margoh', 'Asia'], ['Djourab', 'Africa'], ['Erdi Ennedi', 'Africa'], ['Erg Chech', 'Africa'], ['Erg Igidi', 'Africa'], ['Erg Isaouane', 'Africa'], 
                          ['Erg Maqteir', 'Africa'], ['Erg Ouarane', 'Africa'], ['Erg Rebiana', 'Africa'], ['Ferlo', 'Africa'], ['Fesan', 'Africa'], ['Gibson Desert', 'Australia/Oceania'], ['Gobi', 'Asia'], ['Grand Erg Est', 'Africa'], 
                          ['Grand Erg Ouest', 'Africa'], ['Great Basin', 'America'], ['Great Salt Lake Desert', 'America'], ['Great Sandy Desert', 'Australia/Oceania'], ['Great Victoria Desert', 'Australia/Oceania'], ['Gurbantunggut', 'Asia'], 
                          ['Hamada al-Hamra', 'Africa'], ['Hamada du Draa', 'Africa'], ['Kalahari', 'Africa'], ['Karakum', 'Asia'], ['Kum Tagh', 'Asia'], ['Kysylkum', 'Asia'], ['Libyan Desert', 'Africa'], ['Mojave', 'America'], 
                          ['Mujunkum', 'Asia'], ['Namib', 'Africa'], ['Nefud', 'Asia'], ['Negev', 'Asia'], ['Nubian Desert', 'Africa'], ['Nullarbor Plain', 'Australia/Oceania'], ['Ordos', 'Asia'], ['Owyhee', 'America'], ['Qaidam', 'Asia'], 
                          ['Red Centre', 'Australia/Oceania'], ['Rigestan', 'Asia'], ['Rub Al Chali', 'Asia'], ['Ryn', 'Asia'], ['Saguia el-Hamra', 'Africa'], ['Simpson Desert', 'Australia/Oceania'], ['Sonora', 'America'], 
                          ['Sturt Desert', 'Australia/Oceania'], ['Syrian Desert', 'Asia'], ['TaklaMakan', 'Asia'], ['Talak', 'Africa'], ['Tanami', 'Australia/Oceania'], ['Tanezrouft', 'Africa'], ['Tenere', 'Africa'], 
                          ['Thar', 'Asia'], ['Trarza', 'Africa'], ['Ust Urt', 'Asia']])
    desert = pd.read_csv("./databases/mondial_geo/desert.csv")
    continent = pd.read_csv("./databases/mondial_geo/continent.csv")
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The main part of the desert is in the continent.", 
                           left_df=desert, 
                           right_df=continent, 
                           left_on='Name',
                           right_on='Name',
                           thinking=args.thinking)
    pred = merged_df[['left_Name', 'right_Name']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match116(args):
    query = "Match islands with a province value greater than 50,000 to the continents where their main part is located."
    truth = pd.DataFrame([['Baffin Island', 'America'], ['Banks Island', 'America'], ['Borneo', 'Asia'], ['Cuba', 'America'], ['Cuba', 'Australia/Oceania'], 
                          ['Ellesmere Island', 'America'], ['Great Britain', 'Australia/Oceania'], ['Great Britain', 'Europe'], ['Hispaniola', 'America'], 
                          ['Hokkaido', 'Asia'], ['Honshu', 'Asia'], ['Ireland', 'Europe'], ['Java', 'America'], ['Java', 'Asia'], ['Luzon', 'Asia'], ['Mindanao', 'Asia'], 
                          ['Sulawesi', 'Asia'], ['Sumatra', 'Asia'], ['Te Ika-a-Maui (North Island)', 'Australia/Oceania'], ['Te Waka-a-Maui (South Island)', 'Australia/Oceania'], 
                          ['Victoria Island', 'Africa'], ['Victoria Island', 'America']])
    island = pd.read_csv("./databases/mondial_geo/geo_island.csv")
    continent = pd.read_csv("./databases/mondial_geo/continent.csv")
    island = island[island['Province'] > 50000]
    # 18 * 5
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The main part of the island is in the continent.", 
                           left_df=island, 
                           right_df=continent, 
                           left_on='Island',
                           right_on='Name',
                           thinking=args.thinking)
    pred = merged_df[['left_Island', 'right_Name']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match117(args):
    query = "Match each sea to the continents it adjoins."
    truth = pd.DataFrame([['Andaman Sea', 'Asia'], ['Arabian Sea', 'Asia'], ['Arctic Ocean', 'America'], ['Arctic Ocean', 'Asia'], ['Arctic Ocean', 'Europe'], ['Atlantic Ocean', 'Africa'], ['Atlantic Ocean', 'America'], ['Atlantic Ocean', 'Europe'], 
                          ['Baltic Sea', 'Europe'], ['Barents Sea', 'Asia'], ['Barents Sea', 'Europe'], ['Bay of Bengal', 'Asia'], ['Bering Sea', 'America'], ['Bering Sea', 'Asia'], ['Black Sea', 'Asia'], ['Black Sea', 'Europe'], ['Caribbean Sea', 'America'], 
                          ['East China Sea', 'Asia'], ['Gulf of Aden', 'Africa'], ['Gulf of Aden', 'Asia'], ['Gulf of Mexico', 'America'], ['Gulf of Oman', 'Asia'], ['Indian Ocean', 'Africa'], ['Indian Ocean', 'Asia'], ['Indian Ocean', 'Australia/Oceania'], 
                          ['Irish Sea', 'Europe'],  ['Kattegat', 'Europe'], ['Malakka Strait', 'Asia'], ['Mediterranean Sea', 'Africa'], ['Mediterranean Sea', 'Asia'], ['Mediterranean Sea', 'Europe'], ['North Sea', 'Europe'], 
                          ['Norwegian Sea', 'Europe'], ['Pacific Ocean', 'America'], ['Pacific Ocean', 'Asia'], ['Pacific Ocean', 'Australia/Oceania'], ['Persian Gulf', 'Asia'], ['Red Sea', 'Africa'], ['Red Sea', 'Asia'], ['Sea of Azov', 'Asia'], 
                          ['Sea of Azov', 'Europe'], ['Sea of Japan', 'Asia'], ['Sea of Okhotsk', 'Asia'], ['Sibirian Sea', 'Asia'], ['Skagerrak', 'Europe'], ['South China Sea', 'Asia'], ['Sulawesi Sea', 'Asia'], ['Sunda Sea', 'Asia'], ['The Channel', 'Europe'], ['Yellow Sea', 'Asia']])
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    continent = pd.read_csv("./databases/mondial_geo/continent.csv")
    # 36*5
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The sea adjoins the continent.", 
                           left_df=sea, 
                           right_df=continent, 
                           left_on='Name',
                           right_on='Name',
                           thinking=args.thinking)
    pred = merged_df[['left_Name', 'right_Name']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match118(args):
    query = "Match protected heritage areas where most recent 50 incidents happened in the wildlife coexistence dataset to the countries they are located in."
    truth = pd.DataFrame([['Jasper National Park of Canada', 'Canada'], ['Elk Island National Park of Canada', 'Canada'], ['Riding Mountain National Park of Canada', 'Canada'], ['Banff National Park of Canada', 'Canada'], ['Glacier National Park of Canada', 'Canada'], ['Kootenay National Park of Canada', 'Canada']])
    cihr = pd.read_csv("./databases/santos/cihr_co-applicant_200607.csv")
    wildlife = pd.read_csv("./databases/santos/pca-human-wildlife-coexistence-animals-involved-detailed-records.csv")
    
    wildlife = wildlife.sort_values(by='Incident Date', ascending=False)[:50]
    wildlife = wildlife[['Protected Heritage Area']].drop_duplicates()
    cihr= cihr[['CountryEN']].drop_duplicates()

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The protected heritage area location is in the country.", 
                           left_df=wildlife, 
                           right_df=cihr, 
                           left_on='Protected Heritage Area',
                           right_on='CountryEN',
                           thinking=args.thinking)
    pred = merged_df[['left_Protected Heritage Area', 'right_CountryEN']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match119(args):
    query = "Match cultural spaces with more than 1,000 seats to university-owned public art within a 15-minute walk."
    truth = pd.DataFrame([['Queen Elizabeth Theatre', 113], ['UBC Thunderbird Arena', 127], ['UBC Thunderbird Arena', 132], ['UBC Thunderbird Arena', 133], ['Chan Centre for The Performing Art - Concert Hall', 127], 
                          ['Chan Centre for The Performing Art - Concert Hall', 132], ['Chan Centre for The Performing Art - Concert Hall', 133], ['Orpheum Theatre', 113], ['St Andrews-Wesley United Church', 113], 
                          ['Vogue Theatre', 113], ['Rogers Arena', 113], ['St. Andrews-Wesley United Church', 113], ['Chan Centre for The Performing Art, The- Concert Hall', 127], 
                          ['Chan Centre for The Performing Art, The- Concert Hall', 132], ['Chan Centre for The Performing Art, The- Concert Hall', 133]])
    cultural = pd.read_csv("./databases/nextiaJD/cultural-spaces.csv")
    art = pd.read_csv("./databases/nextiaJD/public-art.csv")
    cultural = cultural[cultural['NUMBER_OF_SEATS'] > 1000]
    cultural = cultural[['CULTURAL_SPACE_NAME', 'Geom']].drop_duplicates()
    art = art[art['Ownership'] == 'university']
    
    print(len(cultural), len(art))
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The two addresses are within a 15-minute walk.", 
                           left_df=cultural, 
                           right_df=art, 
                           left_on='Geom',
                           right_on='Geom',
                           thinking=args.thinking)
    pred = merged_df[['left_CULTURAL_SPACE_NAME', 'right_RegistryID']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()



def pipeline_match120(args):
    query = "Match community centres to cultural spaces with more than 1,000 seats that are within a 5-minute walk."
    truth = pd.DataFrame([['Hastings', 'PNE Forum'], ['Hastings', 'PNE Amphitheatre'], ['Gathering Place Community Centre', 'Orpheum Theatre'], 
                          ['Gathering Place Community Centre', 'Vogue Theatre']])
    centre = pd.read_csv("./databases/nextiaJD/community-centres.csv")
    cultural = pd.read_csv("./databases/nextiaJD/cultural-spaces.csv")

    cultural = cultural[cultural['NUMBER_OF_SEATS'] > 1000]
    cultural = cultural[['CULTURAL_SPACE_NAME', 'Geom']].drop_duplicates()
    print(len(cultural), len(centre))

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The two addresses are within a 5-minute walk.", 
                           left_df=centre, 
                           right_df=cultural, 
                           left_on='Geom',
                           right_on='Geom',
                           thinking=args.thinking)
    pred = merged_df[['left_NAME', 'right_CULTURAL_SPACE_NAME']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match121(args):
    query = "Match suppliers from the April 2018 expenditure dataset to the same suppliers in the May 2015 expenditure dataset."
    truth = pd.DataFrame([['Nhs Supply Chain', 'NHS SUPPLY CHAIN'], ['Novartis Pharmaceuticals Uk Ltd', 'NOVARTIS PHARMACEUTICALS UK LTD'], 
                          ['Roche Diagnostics Limited', 'ROCHE PRODUCTS LTD'], ['Csc', 'CSC COMPUTER SCIENCES LTD'],['Philips Healthcare', 'PHILIPS HEALTHCARE'], ['Nhs Blood And Transplant', 'NHS BLOOD & TRANSPLANT'], ['NHS Litigation Authority', 'NHS LITIGATION AUTHORITY']])
    left = pd.read_csv("./databases/santos/01.Apr_2018.csv")
    right = pd.read_csv("./databases/santos/2015_05_expenditure.csv")
    left = left[['Supplier']].drop_duplicates()
    right = right[['Supplier']].drop_duplicates()
    print(len(left), len(right))

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The two suppliers are the same supplier.", 
                           left_df=left, 
                           right_df=right, 
                           left_on='Supplier',
                           right_on='Supplier',
                           thinking=args.thinking)
    print(merged_df.values.tolist())
    pred = merged_df
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match122(args):
    query = "Find pairs of staff members in the SCS Staff Salaries dataset who share the same first name but are different people."
    truth = pd.DataFrame([['David Ashbourne', 'David Pimm'], ['David Pimm', 'David Ashbourne']])
    staff = pd.read_csv("./databases/nextiaJD/SCS_Staff_Salaries_data_30th_June 2010.csv")

    # print(len(staff), len(authors))
    print(staff)
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                           condition="The two staff have the same first name but are not the same person.", 
                           left_df=staff, 
                           right_df=staff, 
                           left_on='Name',
                           right_on='Name',
                           thinking=args.thinking)
    print(merged_df)
    print(op.get_tokens())
    pred = merged_df[['left_Name', 'right_Name']]
    print(pred.values.tolist())

    return f1_score(truth, pred), op.get_tokens()

def pipeline_match123(args):
    query = "Match US customers to offices located in the same state as the customer's city."
    truth = pd.DataFrame([[124, 1], [129,  1], [131,  3], [151, 3], [161, 1], [173, 2], [175, 2], [181, 3], [205, 1], 
             [219, 1], [239, 1], [286, 2], [319, 3], [320, 2], [321, 1], [347, 1], [362, 2], [424, 3], 
             [447, 1], [450, 1], [456, 3], [462, 2], [475, 1], [495, 2]])

    customers = pd.read_csv("./databases/car_retails/customers.csv")
    customers = customers[customers['country'] == 'USA']
    offices = pd.read_csv("./databases/car_retails/offices.csv")
    print(len(customers), len(offices))
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                        condition="The cities are in the same state", 
                        left_df=customers, 
                        right_df=offices, 
                        left_on='city',
                        right_on='city',
                        thinking=args.thinking)
    pred = merged_df[['left_customerNumber', 'right_officeCode']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match124(args):
    query = "Match non-US customers with a credit limit greater than 100,000 to non-US offices located in cities with the same climate type."
    truth = pd.DataFrame([[114, 4], [114, 7], [119, 4], [119, 7], [141, 4], [146, 4], [187, 4], [187, 7], [227, 4], [227, 7], [249, 5], [249, 6], [259, 4], [259, 7], [276, 5], [276, 6], [278, 5], [278, 6], [298, 4], [298, 7], [386, 5], [386, 6], [458, 4], [496, 4]])
    customers = pd.read_csv("./databases/car_retails/customers.csv")
    customers = customers[(customers['creditLimit'] > 100000) & (customers['country'] !='USA')]
    offices = pd.read_csv("./databases/car_retails/offices.csv")
    offices = offices[offices['country'] !='USA']

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                        condition="The 'left_city' and 'right_city' have the same climate type", 
                        left_df=customers, 
                        right_df=offices, 
                        left_on='city',
                        right_on='city',
                        thinking=args.thinking)
    pred = merged_df[['left_customerNumber', 'right_officeCode']]
    # print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match125(args):
    query = "Match procurement merchant categories from the purchasing card spend dataset to highly similar categories in the rm-mr-2009-eng dataset."
    truth = pd.DataFrame([ ['Human Resources  ', 'Human Resources and Skills Development'], ['Environmental Services  ', 'Environment']])
    merchant = pd.read_csv("./databases/santos/purchasing-card-spend-jul2015.csv")
    rm = pd.read_csv("./databases/santos/rm-mr-2009-eng.csv")
    merchant = merchant[['Procurement (Merchant) Category']].drop_duplicates()
    rm = rm[['MINE']].drop_duplicates()

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                        condition="The two categories are highly similar", 
                        left_df=merchant, 
                        right_df=rm, 
                        left_on='Procurement (Merchant) Category',
                        right_on='MINE',
                        thinking=args.thinking)
    print(merged_df)
    print(op.get_tokens())
    print(merged_df[['left_Procurement (Merchant) Category', 'right_MINE']].values.tolist())
    pred = merged_df[['left_Procurement (Merchant) Category', 'right_MINE']]
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match126(args):
    query = "Match expense types from the IPO payments dataset to highly similar procurement merchant categories in the purchasing card spend dataset."
    truth = pd.DataFrame([['Books Newspapers & Publications', 'Education Library Books '], ['Computer Capital', 'Information Communication Technology  '],  ['Courier service', 'Mail ServicesÃ\x82Â\xa0  '], ['Furniture & Fittings', 'Furniture & Soft Furnishings  '], 
                          ['Law Reporting', 'Legal Services  '], ['Minor Building Works & Maintenance', 'Works - Construction, Repair & Maintenance  '], ['Other Stationery & General Supplies', 'Stationery    '], ['Postage', 'Mail ServicesÃ\x82Â\xa0  '], 
                          ['Seminars and Conferences', 'Human Resources Training & Conferences '],  ['Training', 'Human Resources Training & Conferences '], ['Travel and subsistence Overseas', 'Human Resources Travel & Subsistence '], 
                          ['Travel and subsistence UK', 'Human Resources Travel & Subsistence '], ['Payroll Costs', 'Human Resources  '], ['Recruitment Advertising', 'Human Resources  '], ['Van Delivery Service', 'Mail ServicesÃ\x82Â\xa0  '], ['BIS Solicitors', 'Legal Services  '],
                          ['Books  Newspapers & Publications', 'Education Library Books '], ['Fuel &Utilities', 'Vehicle Management Fuel '], ['Office Cleaning', 'Facilities & Management Services  '], ['Security Guards', 'Facilities & Management Services  '], ['Treasury Solicitors', 'Legal Services  ']])
    pay = pd.read_csv("./databases/santos/1004ipopayments.csv")
    merchant = pd.read_csv("./databases/santos/purchasing-card-spend-jul2015.csv")
    pay = pay[['Expense type']].drop_duplicates()
    merchant = merchant[['Procurement (Merchant) Category']].drop_duplicates()
    print(len(pay), len(merchant))
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                    condition="The two categories are highly similar", 
                    left_df=pay, 
                    right_df=merchant, 
                    left_on='Expense type',
                    right_on='Procurement (Merchant) Category',
                    thinking=args.thinking)
    print(merged_df[['left_Expense type','right_Procurement (Merchant) Category']].values.tolist())
    pred = merged_df[['left_Expense type','right_Procurement (Merchant) Category']]
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match127(args):
    query = "Match cinemas to arts centres located in adjacent wards in the UK."
    truth = pd.DataFrame([['Cineworld Wandsworth', 'Southbank Centre '], ['Picturehouse Central', 'Battersea Arts Centre '], ['Picturehouse Central', 'Southbank Centre '], ['Picturehouse Central', 'The Tabernacle'], 
                          ['Deptford Cinema', 'Dugdale  '], ['Deptford Cinema', 'The Albany '], ['Vue Action', 'Irish Cultural Centre '], ['Vue Action', 'Riverside Studios '], ['Curzon Chelsea', 'The Tabernacle'], 
                          ['Curzon Chelsea', 'The Landmark Arts Centre '], ['The Castle Cinema', 'Dugdale  '], ['The Castle Cinema', 'Rich Mix '], ['The Castle Cinema', 'Poplar Union '], ['The Castle Cinema', 'Fairfield Halls '], 
                          ["Vue Shepherd's Bush\r\n\r\n\r\n\r\n", 'Irish Cultural Centre '], ["Vue Shepherd's Bush\r\n\r\n\r\n\r\n", 'Riverside Studios '], ['Vue Piccadilly', 'Battersea Arts Centre '], ['Vue Piccadilly', 'Southbank Centre '], 
                          ['Vue Piccadilly', 'The Tabernacle'], ['Cineworld Wood Green', 'ICA'], ['Curzon Victoria', 'Battersea Arts Centre '], ['Curzon Victoria', 'Southbank Centre '], ['Curzon Victoria', 'The Tabernacle'], 
                          ['Cineworld Fulham Road', 'The Tabernacle'], ['Curzon Goldsmiths', 'Dugdale  '], ['Curzon Goldsmiths', 'The Albany '], ['Everyman Hampstead', 'Dugdale  '], ['Everyman Hampstead', 'Jacksons Lane '], 
                          ['Everyman Hampstead', 'JW3'], ['Odeon Swiss Cottage', 'Dugdale  '], ['Odeon Swiss Cottage', 'JW3'], ['Odeon Beckenham', 'Dugdale  '], ['Odeon Beckenham', 'Redbridge Drama Centre '], ['Imperial College Union Cinema', 'ICA'], 
                          ['David Lean Cinema', 'Redbridge Drama Centre '], ['David Lean Cinema', 'The Albany '], ['David Lean Cinema', 'The Landmark Arts Centre '], ['David Lean Cinema', 'Stratford Circus Arts Centre '], ['Odeon Streatham', 'Battersea Arts Centre '], 
                          ['Odeon Streatham', 'Dugdale  '], ['Odeon Streatham', 'Southbank Centre '], ['Odeon Streatham', 'Streatham Space Project '], ['Odeon Tottenham Court Road', 'ICA'], ['Covent Garden Hotel', 'Battersea Arts Centre '], 
                          ['Covent Garden Hotel', 'ICA'], ['Vue North Finchley', 'Arts Depot '], ['Curzon Bloomsbury', 'ICA']])
    cinemas = pd.read_csv("./databases/santos/cinemas.csv")
    arts = pd.read_csv("./databases/santos/Arts_centres.csv")
    # cinemas = cinemas[['ward_2018_name']].drop_duplicates()
    # arts = arts[['ward_2018_name']].drop_duplicates()
    print(len(cinemas), len(arts))
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                    condition="The two buildings are in adjacent wards in UK", 
                    left_df=cinemas, 
                    right_df=arts, 
                    left_on='ward_2018_name',
                    right_on='ward_2018_name',
                    thinking=args.thinking)
    print(merged_df[['left_name', 'right_name']].values.tolist())
    pred = merged_df[['left_name', 'right_name']]
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match128(args):
    query = "Match arts centres to community centres in Islington whose website addresses have an edit distance of no more than 10."
    truth = pd.DataFrame([['Jacksons Lane ', 'Hilldrop Community Centre'], ['JW3', 'Hilldrop Community Centre'], ['The Albany ', 'Hilldrop Community Centre'], ['The Barbican ', 'Hilldrop Community Centre'], ['Watermans Arts Centre', 'Hilldrop Community Centre']])
    arts = pd.read_csv("./databases/santos/Arts_centres.csv")
    community = pd.read_csv("./databases/santos/community_centres.csv")
    community = community[community['borough_name']=='Islington']
    community = community.dropna(subset=['website'])
    arts = arts.dropna(subset=['website'])

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                    condition="The edit distance between two websites are no more than 10", 
                    left_df=arts, 
                    right_df=community, 
                    left_on='website',
                    right_on='website',
                    thinking=args.thinking)
    print(merged_df)
    print(op.get_tokens())
    print(merged_df[['left_website', 'right_website']])
    pred = merged_df[['left_name', 'right_name']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match129(args):
    query = "Match cities with a population greater than 5,000,000 to their closest sea."
    truth = pd.DataFrame([['Bangkok', 'Bay of Bengal'], ['Beijing', 'Yellow Sea'], ['Bogota', 'Caribbean Sea'], ['Cairo', 'Mediterranean Sea'], ['Hong Kong', 'South China Sea'], ['Istanbul', 'Black Sea'], 
                          ['Jakarta', 'Sunda Sea'], ['Karachi', 'Arabian Sea'], ['Lagos', 'Atlantic Ocean'], ['Lahore', 'Arabian Sea'], ['Lima', 'Pacific Ocean'], ['London', 'North Sea'], 
                          ['Mexico City', 'Gulf of Mexico'], ['Moscow', 'Baltic Sea'], ['Mumbai', 'Arabian Sea'], ['New Delhi', 'Arabian Sea'], ['New York', 'Atlantic Ocean'], ['Rio de Janeiro', 'Atlantic Ocean'], 
                          ['Sao Paulo', 'Atlantic Ocean'], ['Seoul', 'Yellow Sea'], ['Shanghai', 'East China Sea'], ['Tehran', 'Persian Gulf'], ['Tianjin', 'Yellow Sea'], ['Tokyo', 'Pacific Ocean']])
    city = pd.read_csv("./databases/mondial_geo/city.csv")
    city = city[city['Population'] > 5000000]
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=args.impl, 
                    condition="Among all provided seas, map each city to its closest sea.", 
                    left_df=city, 
                    right_df=sea, 
                    left_on='Name',
                    right_on='Name',
                    thinking=args.thinking)
    print(merged_df[['left_Name', 'right_Name']].values.tolist())
    pred = merged_df[['left_Name', 'right_Name']]
    return f1_score(truth, pred), op.get_tokens()

# fuzzy and semantic, medium
def pipeline_match200(args):
    query = "Match restaurants in zip code 60642 between the Yelp and Zomato datasets that refer to the same restaurant."
    truth = pd.DataFrame([
        (2, 844), 
        (72, 1020),
        (90, 744),
        (111, 564),
        (198, 1022),
        (275, 272), 
        (333, 134),
        (418, 590)
    ])
    yelp = pd.read_csv("./databases/restaurants2/yelp.csv")
    zomato = pd.read_csv("./databases/restaurants2/zomato.csv")
    yelp = yelp[yelp['zip'] == 60642]
    zomato = zomato[zomato['zip'] == 60642]
    # print(yelp[['ID', 'name', 'address']])
    # print(zomato[['ID', 'name', 'address']])
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same restaurant",
                            left_df=yelp,
                            right_df=zomato,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy and reasoning, medium
def pipeline_match201(args):
    query = "Match restaurants in zip code 60616 between the Yelp and Zomato datasets that refer to the same restaurant."
    truth = pd.DataFrame([
        (36, 1109),
        (86, 763),
        (546, 929),
        (589, 1272),
        (404, 1303),
        (650, 163)
    ])
    yelp = pd.read_csv("./databases/restaurants2/yelp.csv")
    zomato = pd.read_csv("./databases/restaurants2/zomato.csv")
    yelp = yelp[yelp['zip'] == 60616]
    zomato = zomato[zomato['zip'] == 60616]
    # print(yelp[['ID', 'name', 'address']])
    # print(zomato[['ID', 'name', 'address']])
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same restaurant",
                            left_df=yelp,
                            right_df=zomato,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy, easy
def pipeline_match202(args):
    query = "Match highly-rated movies (IMDb rating > 8.3 and Rotten Tomatoes rating > 8.5) between the IMDb and Rotten Tomatoes datasets that refer to the same movie."
    truth = pd.DataFrame([
        ('a-456', 'b-434'),
        ('a-569', 'b-534'),
        ('a-701', 'b-653'),
    ])
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    tomato = pd.read_csv("./databases/movies/rotten_tomatoes.csv")
    imdb = imdb[pd.to_numeric(imdb['Rating'], errors='coerce') > 8.3]
    tomato = tomato[pd.to_numeric(tomato['Rating'], errors='coerce') > 8.5]
    # print(imdb[['ID', 'Title']])
    # print(tomato[['ID', 'Title']])
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same movie",
                            left_df=imdb,
                            right_df=tomato,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy, easy
def pipeline_match203(args):
    query = "Match poorly-rated movies (rating < 2.3) between the IMDb and Rotten Tomatoes datasets that refer to the same movie."
    truth = pd.DataFrame([
        ('a-708', 'b-662'),
        ('a-1775', 'b-1553')
    ])
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    tomato = pd.read_csv("./databases/movies/rotten_tomatoes.csv")
    imdb = imdb[pd.to_numeric(imdb['Rating'], errors='coerce') < 2.3]
    tomato = tomato[pd.to_numeric(tomato['Rating'], errors='coerce') < 2.3]
    # print(imdb[['ID', 'Title']])
    # print(tomato[['ID', 'Title']])
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same movie",
                            left_df=imdb,
                            right_df=tomato,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# complex reasoning and external knowledge, hard
def pipeline_match204(args):
    query = "Match electronics products originally priced over $2,800 between the Amazon and Best Buy datasets that refer to the same product."
    truth = pd.DataFrame([
        (3367, 102),
        (1514, 798),
        (3150, 1011)
    ])
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    bestbuy =  pd.read_csv("./databases/electronics/best_buy.csv")
    amazon = amazon[pd.to_numeric(amazon['Original_Price'].str.replace(r'[\$,]', '', regex=True)) > 2800]
    bestbuy = bestbuy[pd.to_numeric(bestbuy['Price'].str.replace(r'[\$,]', '', regex=True)) > 2800]
    pd.set_option('display.max_colwidth', None)
    # print(amazon[['ID', 'Name']])
    # print(bestbuy[['ID', 'Name']])
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same product",
                            left_df=amazon,
                            right_df=bestbuy,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# complex reasoning and external knowledge, hard
def pipeline_match205(args):
    query = "Match Dell electronics products priced over $1,500 between the Amazon and Best Buy datasets that have the same configurations."
    truth = pd.DataFrame([
        (3420, 1095),
        (2755, 108),
        (2755, 682)
    ])
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    bestbuy =  pd.read_csv("./databases/electronics/best_buy.csv")
    amazon = amazon[amazon['Brand'] == 'Dell']
    bestbuy = bestbuy[bestbuy['Brand'] == 'Dell']
    amazon = amazon[pd.to_numeric(amazon['Original_Price'].str.replace(r'[\$,]', '', regex=True)) > 1500]
    bestbuy = bestbuy[pd.to_numeric(bestbuy['Price'].str.replace(r'[\$,]', '', regex=True)) > 1500]
    # pd.set_option('display.max_colwidth', None)
    # print(amazon.shape)
    # print(bestbuy.shape)
    # print(amazon[['ID', 'Name']])
    # print(bestbuy[['ID', 'Name']])
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows have the same configurations",
                            left_df=amazon,
                            right_df=bestbuy,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy, easy
def pipeline_match206(args):
    query = "Match songs from the 'Fearless Platinum Edition' album on Amazon Music to the corresponding karaoke versions on iTunes that refer to the same song."
    truth = pd.DataFrame([
        (6675, 17064),
        (6676, 17065),
        (6677, 17066),
        (6678, 17067),
        (6679, 17068),
        (6680, 17069),
        (6681, 17070),
        (6682, 17071),
        (6683, 17072),
        (6684, 17073),
        (6685, 17074),
        (6686, 17075),
        (6687, 17076),
    ])
    amazon = pd.read_csv("./databases/music/amazon_music.csv")
    itunes = pd.read_csv("./databases/music/itunes.csv")
    amazon = amazon[amazon['Album_Name'] == 'Fearless Platinum Edition']
    itunes = itunes[itunes['Album_Name'] == 'Taylor Swift Karaoke: Fearless (Instrumentals With Background Vocals)']
    # print(amazon[['Sno', 'Song_Name']])
    # print(itunes[['Sno', 'Song_Name']])
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same song",
                            left_df=itunes,
                            right_df=amazon,
                            thinking = args.thinking)
    # print(matched_df[['left_Song_Name', 'right_Song_Name']])
    pred = matched_df[['left_Sno', 'right_Sno']]
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy, easy
def pipeline_match207(args):
    query = "Match songs from Taylor Swift's self-titled album on Amazon Music to the corresponding karaoke versions on iTunes that refer to the same song."
    truth = pd.DataFrame([
        (6601, 44114),
        (6602, 44115),
        (6603, 44128),
        (6604, 44117),
        (6605, 44118),
        (6606, 44119),
        (6607, 44120),
        (6608, 44121),
        (6609, 44122),
        (6610, 44123),
        (6611, 44124),
        (6612, 44125),
        (6613, 44126),
        (6614, 44127),
    ])
    amazon = pd.read_csv("./databases/music/amazon_music.csv")
    itunes = pd.read_csv("./databases/music/itunes.csv")
    amazon = amazon[(amazon['Album_Name'] == 'Taylor Swift') & (amazon['Artist_Name'] == 'Taylor Swift')]
    itunes = itunes[itunes['Album_Name'] == 'Taylor Swift Karaoke (Instrumentals With Background Vocals)']
    # print(amazon[['Sno', 'Song_Name']])
    # print(itunes[['Sno', 'Song_Name']])
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same song",
                            left_df=itunes,
                            right_df=amazon,
                            thinking = args.thinking)
    pred = matched_df[['left_Sno', 'right_Sno']]
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy and reasoning, medium
def pipeline_match208(args):
    query = "Match the top 15 highest-rated anime between the Anime Planet and MyAnimeList datasets that refer to the same animation."
    truth = pd.DataFrame([
        (98, 300),
        (97, 300),
        (100, 306),
        (99, 306),
        (101, 301),
        (102, 303),
        (103, 303),
        (104, 308),
        (105, 309), 
        (107, 317), 
        (108, 307),
        (106, 304),
        (113, 305), 
    ])
    planet = pd.read_csv("./databases/anime/anime_planet.csv")
    mylist = pd.read_csv("./databases/anime/my_anime_list.csv")
    planet = planet.sort_values(by='Rating', ascending=False)[:15]
    mylist = mylist.sort_values(by='Rating', ascending=False)[:15]
    # pd.set_option('display.max_colwidth', None)
    # print(planet)
    # print(mylist)
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same animation",
                            left_df=planet,
                            right_df=mylist,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy and reasoning, medium
def pipeline_match209(args):
    query = "Match the top 10 highest-rated 'Conan' anime between the Anime Planet and MyAnimeList datasets that refer to the same animation."
    truth = pd.DataFrame([
        (349, 57),
        (350, 57), 
        (153, 219),
        (155, 219),
        (541, 253),
        (543, 253)
    ])
    planet = pd.read_csv("./databases/anime/anime_planet.csv")
    mylist = pd.read_csv("./databases/anime/my_anime_list.csv")
    planet = planet[planet['Title'].str.contains('Conan')].sort_values(by='Rating', ascending=False)[:10]
    mylist = mylist[mylist['Title'].str.contains('Conan')].sort_values(by='Rating', ascending=False)[:10]
    # pd.set_option('display.max_colwidth', None)
    # print(planet)
    # print(mylist)
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same animation",
                            left_df=planet,
                            right_df=mylist,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match210(args):
    query = "Match 'Clannad' anime between the Anime Planet and MyAnimeList datasets that refer to the same animation."
    truth = pd.DataFrame([[105, 309], [212, 258], [535, 14], [1860, 2105]])
    planet = pd.read_csv("./databases/anime/anime_planet.csv")
    mylist = pd.read_csv("./databases/anime/my_anime_list.csv")
    planet = planet[planet['Title'].str.contains('Clannad')]
    mylist = mylist[mylist['Title'].str.contains('Clannad')]

    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same animation",
                            left_df=planet,
                            right_df=mylist,
                            thinking = args.thinking)
    # pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_columns', None) 
    # print(matched_df)
    pred = matched_df[['left_ID', 'right_ID']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match211(args):
    query = "Match the Naruto-related animations from Anime Planet to those from My Anime List that refer to the same animation."
    truth = pd.DataFrame([[49, 448], [471, 520], [476, 661], [682, 666], [683, 666], [997, 1102], [1477, 853], [1532, 1390], [1533, 1390], [1575, 1353], [1576, 1353], [1798, 1125], [1864, 1651], [3050, 2154], [3129, 1609], [3130, 1609], [3465, 2641], [3607, 2155], [3633, 3372], [3712, 3345]])
    planet = pd.read_csv("./databases/anime/anime_planet.csv")
    mylist = pd.read_csv("./databases/anime/my_anime_list.csv")
    planet = planet[planet['Title'].str.contains('Naruto')]
    mylist = mylist[mylist['Title'].str.contains('Naruto')]
    print(len(planet), len(mylist))
    print(planet)
    print(mylist)

    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same animation",
                            left_df=planet,
                            right_df=mylist,
                            thinking = args.thinking)
    # pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_columns', None) 
    # print(matched_df[['left_Title', 'right_Title']])
    pred = matched_df[['left_ID', 'right_ID']]
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match212(args):
    query = "Match the Fate-related animations from Anime Planet to those from My Anime List that refer to the same animation."
    truth = pd.DataFrame([[59, 140], [59, 150], [73, 150], [93, 225], [124, 122], [124, 328], [253, 122], [253, 2228], [616, 1093], [617, 1093], [868, 1153], [1650, 2129], [1679, 2176], [2120, 2931], [3079, 1423], [3079, 2182], [3132, 1423], [3132, 2129], [3132, 3841], [3655, 2962], [3981, 2129]])
    planet = pd.read_csv("./databases/anime/anime_planet.csv")
    mylist = pd.read_csv("./databases/anime/my_anime_list.csv")
    planet = planet[planet['Title'].str.contains('Fate')]
    mylist = mylist[mylist['Title'].str.contains('Fate')]
    print(len(planet), len(mylist))
    print(planet)
    print(mylist)

    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same animation",
                            left_df=planet,
                            right_df=mylist,
                            thinking = args.thinking)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None) 
    print(matched_df[['left_Title', 'left_Episodes', 'right_Title', 'right_Episodes']])
    pred = matched_df[['left_ID', 'right_ID']]
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()
    
def pipeline_match213(args):
    query = "Match the Christopher Nolan movies from IMDb to those from Rotten Tomatoes that refer to the same movie."
    truth = pd.DataFrame([['a-101', 'b-101'], ['a-362', 'b-347'], ['a-701', 'b-653'], ['a-1279', 'b-1103'], ['a-1820', 'b-1596'], ['a-2318', 'b-2042']])
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    tomato = pd.read_csv("./databases/movies/rotten_tomatoes.csv")
    imdb = imdb[imdb['Director'] == 'Christopher Nolan']
    tomato = tomato[tomato['Director'] == 'Christopher Nolan']
    print(imdb)
    print(tomato)
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same movie",
                            left_df=imdb,
                            right_df=tomato,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    # print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match214(args):
    query = "Match the Clint Eastwood movies from IMDb to those from Rotten Tomatoes that refer to the same movie."
    truth = pd.DataFrame([['a-350', 'b-338'], ['a-571', 'b-536'], ['a-692', 'b-645'], ['a-735', 'b-687'], ['a-970', 'b-904'], ['a-1340', 'b-1161'], ['a-1711', 'b-1495'], ['a-2161', 'b-1906'], ['a-2248', 'b-1982']])
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    tomato = pd.read_csv("./databases/movies/rotten_tomatoes.csv")
    imdb = imdb[imdb['Director'] == 'Clint Eastwood']
    tomato = tomato[tomato['Director'] == 'Clint Eastwood']
    print(imdb)
    print(tomato)
    op = LogicalMatch(operand_type=OperandType.ROW)
    matched_df = op.execute(impl_type=args.impl,
                            condition="The two rows refer to the same movie",
                            left_df=imdb,
                            right_df=tomato,
                            thinking = args.thinking)
    pred = matched_df[['left_ID', 'right_ID']]
    # print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()


# fuzzy, easy
def pipeline_match300(args):
    query = "Match the columns of cultural spaces dataset to the columns of street intersections dataset that refer to the same feature."
    truth = pd.DataFrame([
        ('Geo Local Area', 'LOCAL_AREA'),
        ('Geom', 'Geom'), 
    ])
    culture = pd.read_csv("./databases/nextiaJD/cultural-spaces.csv")
    intersections = pd.read_csv("./databases/nextiaJD/street-intersections.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the same feature",
                        left_df=culture,
                        right_df=intersections, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy and reasoning, medium
def pipeline_match301(args):
    query = "Match the columns of community centres dataset to the columns of community gardens and food trees dataset that refer to the same feature."
    truth = pd.DataFrame([
        ('NAME', 'STREET_NAME'),
        ('ADDRESS', 'MERGED_ADDRESS'), 
        ('URLLINK', 'WEBSITE'),
        ('Geom', 'Geom'),
        ('Geo Local Area', 'Geo Local Area')
    ])
    centres = pd.read_csv("./databases/nextiaJD/community-centres.csv")
    gardens = pd.read_csv("./databases/nextiaJD/community-gardens-and-food-trees.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the same feature",
                        left_df=centres,
                        right_df=gardens, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy and reasoning, medium
def pipeline_match302(args):
    query = "Match the columns of libraries dataset to the columns of public art dataset that refer to the same feature."
    truth = pd.DataFrame([
        ('ADDRESS', 'SiteAddress'),
        ('URLLINK', 'URL'),
        ('Geom', 'Geom'),
        ('Geo Local Area', 'GeoLocalArea')
    ])
    libraries = pd.read_csv("./databases/nextiaJD/libraries.csv")
    art = pd.read_csv("./databases/nextiaJD/public-art.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the same feature",
                        left_df=libraries,
                        right_df=art, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy and reasoning, hard
def pipeline_match303(args):
    query = "Match the columns of Botswana 2011 population census dataset to the columns of population by governorate, citizenship and gender dataset that refer to the same feature."
    truth = pd.DataFrame([
        ('DATE', 'Year'),
        ('SEX_NAME', 'Gender'),
        ('VALUE', 'Population')
    ])
    botswana =  pd.read_csv("./databases/nextiaJD/population-census-of-botswana-2011.csv")
    govern = pd.read_csv("./databases/nextiaJD/population-by-governorate-citizenship-and-gender.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the same feature",
                        left_df=botswana,
                        right_df=govern, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# hard
def pipeline_match304(args):
    query = "Match the columns of cultural spaces dataset to the columns of public art dataset that refer to the same feature."
    truth = pd.DataFrame([
        ('YEAR',     'YearOfInstallation'),
        ('CULTURAL_SPACE_NAME', 'SiteName'),
        ('WEBSITE', 'URL'),
        ('ADDRESS', 'SiteAddress'),
        ('LOCAL_AREA', 'GeoLocalArea'),
        ('OWNERSHIP', 'Ownership'),
        ('Geom', 'Geom')
    ])
    culture = pd.read_csv("./databases/nextiaJD/cultural-spaces.csv")
    art = pd.read_csv("./databases/nextiaJD/public-art.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the same feature",
                        left_df=culture,
                        right_df=art, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# TODO: doubke check
# hard
def pipeline_match305(args):
    query = "Match the columns of the FRPM dataset to the columns of the schools dataset that refer to the same feature."
    truth = pd.DataFrame([
        ('CDSCode',     'CDSCode'),
        ('County Code', 'County'),
        ('District Code', 'NCESDist'),
        ('District Name', 'District'),
        ('School Name', 'School'),
        ('District Type ','DOCType'),
        ('School Type','SOCType'),
        ('Educational Option Type',    'EdOpsName'),
        ('Charter School (Y/N)' ,     'Charter'),
        ('Charter School Number' ,  'CharterNum'),
        ('Charter Funding Type' , 'FundingType'),
    ])
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")
    schools = pd.read_csv("./databases/california_schools/schools.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the same feature",
                        left_df=frpm,
                        right_df=schools, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    return f1_score(truth, pred), op.get_tokens()
    

# fuzzy, easy
def pipeline_match306(args):
    query = "Match the columns of the Amazon electronics dataset to the columns of the Best Buy dataset that refer to the same feature."
    truth = pd.DataFrame([
        ('ID',     'ID'),
        ('Brand', 'Brand'),
        ('Amazon_Price', 'Price'),
        ('Original_Price', 'Price'),
        ('Features', 'Features'),
    ])
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    bestbuy =  pd.read_csv("./databases/electronics/best_buy.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the same feature",
                        left_df=amazon,
                        right_df=bestbuy, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    # print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# medium
def pipeline_match307(args):
    query = "Match the columns of the schools dataset to the columns of the EO4 dataset that refer to the similar type of information."
    truth = pd.DataFrame([
            ('ADDRESS',     'STREET'),
            ('SCHOOL_NAME', 'NAME'),
    ])
    schools = pd.read_csv("./databases/nextiaJD/schools.csv")
    eo4 = pd.read_csv("./databases/nextiaJD/eo4.csv", delimiter=',')
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the similar type of information",
                        left_df=schools,
                        right_df=eo4, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

# easy
def pipeline_match308(args):
    query = "Match the columns of the authors dataset to the columns of the public art artists dataset that refer to the similar type of information."
    truth = pd.DataFrame([
            ('Author_ID',     'ArtistID'),
            ('NAME', 'FirstName'),
            ('NAME', 'LastName'),
    ])
    authors = pd.read_csv("./databases/nextiaJD/datasets_579296_1047868_authors.csv", delimiter=',')
    artists = pd.read_csv("./databases/nextiaJD/public-art-artists.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the similar type of information",
                        left_df=authors,
                        right_df=artists, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

def pipeline_match309(args):
    query = "Match the columns of the community gardens and food trees dataset to the columns of the rental standards current issues dataset that refer to the similar feature."
    truth = pd.DataFrame([
            ('NAME',     'BUSINESSOPERATOR'),
            ('WEBSITE', 'DetailURL'),
            ('STREET_NAME', 'StreetNumber'),
            ('STREET_NUMBER', 'Street'),
            ('Geo Local Area', 'Geo Local Area'),
            ('Geom', 'Geom'),
    ])
    gardens = pd.read_csv("./databases/nextiaJD/community-gardens-and-food-trees.csv")
    rental = pd.read_csv("./databases/nextiaJD/rental-standards-current-issues.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the silimar feature",
                        left_df=gardens,
                        right_df=rental, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    print(pred)
    return f1_score(truth, pred), op.get_tokens()
    

def pipeline_match310(args):
    query = "Match the columns of the state expenditures dataset to the columns of the RM-MR 2009 dataset that refer to the similar feature."
    truth = pd.DataFrame([['Fiscal Year', 'Ã\x83Â¯Ã\x82Â»Ã\x82Â¿FSCL_YR'], ['Agency Name', 'DEPT_EN_DESC']])
    state = pd.read_csv("./databases/santos/state_expenditures.csv")
    rm_2009 = pd.read_csv("./databases/santos/rm-mr-2009-eng.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the silimar feature",
                        left_df=state,
                        right_df=rm_2009, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match311(args):
    query = "Match the columns of the May 2018 spending dataset to the columns of the April 2018 spending dataset that refer to the similar feature."
    truth = pd.DataFrame([['ORGANISATION NAME', 'Department Family'], ['EFFECTIVE DATE', 'Date'], ['DATE PAID', 'Date'], ['SUPPLIER NAME', 'Supplier'], ['TRANSACTION NO.', 'Transaction Number'], ['AMOUNT (Ã\x82Â£)', 'Amount']])
    may = pd.read_csv("./databases/santos/2018-may-spend-over-500.csv")
    april = pd.read_csv("./databases/santos/01.Apr_2018.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the silimar feature",
                        left_df=may,
                        right_df=april, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()


def pipeline_match312(args):
    query = "Match the columns of the state expenditures dataset to the columns of the AMO-AME 2007 dataset that refer to the similar feature."
    truth = pd.DataFrame([['Fiscal Year', 'Ã\x83Â¯Ã\x82Â»Ã\x82Â¿FSCL_YR'], ['Agency Name', 'DEPT_EN_DESC'], ['Payments Total', 'AGRG_PYMT_AMT']])
    amo = pd.read_csv("./databases/santos/amo-ame-2007-eng.csv")
    state = pd.read_csv("./databases/santos/state_expenditures.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the silimar feature",
                        left_df=state,
                        right_df=amo, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match313(args):
    query = "Match the columns of the home office senior officials travel dataset to the columns of the travel expenses April-June 2018 dataset that refer to the similar feature."
    truth = pd.DataFrame([['Name of Official', 'Senior Officials Name'], ['Mode of travel', 'Mode of transport'], ['Destination', 'Destination'], ['Class of travel', 'Class of travel'], [' Base Sales including tax ', 'Total cost, including all visas, accommodation, travel, meals etc. (Â£)']])
    home = pd.read_csv("./databases/santos/home_office_senior_officials_travel_data_return.csv")
    travel = pd.read_csv("./databases/santos/travel-exp-April-June-2018.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the silimar feature",
                        left_df=home,
                        right_df=travel, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

def pipeline_match314(args):
    query = "Match the columns of the CIHR co-applicant dataset to the columns of the cabinet office senior officials dataset that refer to the similar feature."
    truth = pd.DataFrame([['CoAppNm', 'Name'], ['CoAppOrgNm', 'Organisation'], ['ProvinceEN', 'Unit']])
    cihr = pd.read_csv("./databases/santos/cihr_co-applicant_200607.csv")
    cabinet = pd.read_csv("./databases/santos/cabinet-office__30-09-2015__cabinetoffice_CO-Template-FINAL-senior.csv")
    op = LogicalMatch(operand_type=OperandType.COLUMN)
    pred = op.execute(impl_type=args.impl,
                        condition="The two columns refer to the silimar feature",
                        left_df=cihr,
                        right_df=cabinet, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    print(pred.values.tolist())
    return f1_score(truth, pred), op.get_tokens()

if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(impl=ImplType.LLM_ONE, thinking=True, example_num=3)
    print(pipeline_match300(args))
