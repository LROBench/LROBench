import os
import sys
import csv
import time
import pandas as pd
import multiprocessing

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import (
    LogicalSelect,
    LogicalMatch,
    LogicalImpute,
    LogicalCluster,
    LogicalOrder,
)
from src.core.enums import OperandType, ImplType

# shipping
# select + select
def pipeline_1(args):
    query = "How many shipments did northeast USA customers make before the opening ceremony of the 2016 Olympics?"
    answer = 12
    shipment = pd.read_csv("./databases/shipping/shipment.csv")
    customer = pd.read_csv("./databases/shipping/customer.csv")
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    customer = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The customer is located in the northeast side of USA",
        df=customer,
        depend_on=["city", "state"],
        thinking=True,
    )
    df = pd.merge(shipment, customer, on="cust_id")
    
    df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The shipment happened before the opening ceremony of the 2016 Olympics",
        df=df,
        depend_on=["ship_date"],
        thinking=True,
    )
    
    prediction = df.shape[0]  
    return prediction, op.total_tokens


# restaurants2
# select + select + impute
def pipeline_2(args):
    query = "Infer the country of origin of asian cuisines served at popular Yelp restaurants (votes > 3000) in the Northeastern city of the United States?"
    answer = [['China'], ['Japan'], ['Japan']]
    yelp = pd.read_csv("./databases/restaurants2/yelp.csv")
    yelp = yelp[yelp["votes"] > 3000]
    print(yelp)
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    yelp = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The zip belongs to a northeast city of USA",
        df=yelp,
        depend_on=["zip"],
        thinking=True,
    )
    
    print(yelp)
    
    yelp = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The cuisine exactly originates from Asia.",
        df=yelp,
        depend_on=["cuisine"],
        thinking=True,
    )
    print(len(yelp))
    print(yelp.columns)
    print(yelp)

    op2 = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Impute the cuisine's country of origin.",
        df=yelp,
        depend_on="cuisine",
        new_col="ori_region",
        thinking = True,
    )
    print(result)
    prediction = result[['ori_region']].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# superhero
# select + select
def pipeline_3(args):
    query = "List DC Comics superheroes that are taller than Michael Jordan, have red eyes, and possess a power related to agility?"
    answer = "['Amazo', 'Ares', 'Darkseid', 'Doomsday', 'Killer Croc', 'Martian Manhunter']"

    hero = pd.read_csv("./databases/superhero/superhero.csv")
    publisher = pd.read_csv("./databases/superhero/publisher.csv")
    power = pd.read_csv("./databases/superhero/superpower.csv")
    hero_power = pd.read_csv("./databases/superhero/hero_power.csv")
    colour = pd.read_csv("./databases/superhero/colour.csv")

    publisher = publisher[publisher["publisher_name"] == "DC Comics"]
    colour = colour[colour["colour"] == "Red"]

    merged_df = pd.merge(
        hero.rename(columns={"id": "hero_id"}),
        publisher.rename(columns={"id": "publisher_id"}),
        on="publisher_id",
    )
    merged_df = pd.merge(merged_df, hero_power, on="hero_id")
    merged_df = pd.merge(
        merged_df, power.rename(columns={"id": "power_id"}), on="power_id"
    )
    merged_df = pd.merge(
        merged_df,
        colour.rename(columns={"id": "eye_colour_id", "colour": "eye_colour"}),
        on="eye_colour_id",
    )

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This power is related to agility.",
        df=merged_df,
        depend_on=["power_name"],
        thinking = True,
    )

    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This hero is higher than Michael Jordan.",
        df=result,
        depend_on=["height_cm"],
        thinking=True,
    )
    prediction = result[["superhero_name"]].drop_duplicates().values.tolist()
    
    return prediction, op.total_tokens


# california_schools
# select + order
def pipeline_4(args):
    query = "List all counties that have an Elementary school with an Percent (%) Eligible FRPM (K-12) less than 1%, and sort them by population in descending order."
    answer = ["Los Angeles", "Contra Costa", "Marin", "Santa Clara", "San Mateo", "Alameda"]

    schools = pd.read_csv("./databases/california_schools/schools.csv")
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")

    merged_df = pd.merge(schools, frpm, on="CDSCode")
    merged_df = merged_df.dropna(subset=["FRPM Count (K-12)"])
    merged_df = merged_df[merged_df["Percent (%) Eligible FRPM (K-12)"] < 0.01]

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    result = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This school is an Elementary school.",
        df=merged_df,
        depend_on=["School Name"],
        thinking=True,
    )

    result = result[["County"]].drop_duplicates()
    op2 = LogicalOrder(operand_type=OperandType.ROW)
    k = len(result)
    prediction = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="the population of county listed in the 'County' column",
        depend_on="County",
        ascending=False,
        k=k,
        df=result,
        thinking=True,
    )
    prediction = prediction["County"].tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# electronics
# select + select
def pipeline_5(args):
    query = "Among Top 10 expensive electronic products sold on Amazon, how many of their brands originate outside of the USA and has a 8GB memory?"
    answer = 4

    amazon_product = pd.read_csv("./databases/electronics/amazon.csv")

    amazon_product = amazon_product.dropna(subset=["Amazon_Price"])
    amazon_product = amazon_product.sort_values(by="Amazon_Price", ascending=False)[:10]
    
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    result = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This product's brand originates outside USA.",
        df=amazon_product,
        depend_on=["Brand"],
        thinking=True,
    )

    result = result[["ID", "Features"]]

    op2 = LogicalSelect(operand_type=OperandType.ROW)
    result = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This product's has 8GB memory.",
        df=result,
        depend_on=["Features"],
        thinking=True,
    )

    prediction = len(result[["ID"]])
    return prediction, op1.total_tokens + op2.total_tokens


# books
# select + select
def pipeline_6(args):
    query = "How many books are written in a language originates from an Asian country, and are bought by customers using a government affiliated email?"
    answer = 3
    customer = pd.read_csv("./databases/books/customer.csv")
    cust_order = pd.read_csv("./databases/books/cust_order.csv")
    book = pd.read_csv("./databases/books/book.csv")
    book_language = pd.read_csv("./databases/books/book_language.csv")
    order_line = pd.read_csv("./databases/books/order_line.csv")
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    book_language = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This language originates from Asia.",
        df=book_language,
        depend_on=["language_name"],
        thinking = True,
    )
    print(book_language)

    merged_df = pd.merge(customer, cust_order, on="customer_id")
    merged_df = pd.merge(merged_df, order_line, on="order_id")
    merged_df = pd.merge(merged_df, book, on="book_id")
    merged_df = pd.merge(merged_df, book_language, on="language_id")
    
    print(merged_df.columns)
    print(merged_df)
    
    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This is a government affiliated email.",
        df=merged_df,
        depend_on=["email"],
        thinking = True,
    )
    print(merged_df)
    merged_df = merged_df[["book_id"]].drop_duplicates()
    
    prediction = len(merged_df)
    return prediction, op.total_tokens


# formula_1
# select + select
def pipeline_7(args):
    query = "How many drivers born in Europe and after 1995 have been finished the race without any disqualification and damage, and with a gap of less than 2 Laps compared with the race winner."
    answer = 2

    results = pd.read_csv("./databases/formula_1/results.csv")
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")
    status = pd.read_csv("./databases/formula_1/status.csv")

    drivers = drivers[pd.to_datetime(drivers["dob"]).dt.year > 1995]
    op = LogicalSelect(operand_type=OperandType.ROW)
    drivers = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This Formula 1 driver is born in Europe.",
        df=drivers,
        depend_on=["nationality"],
        thinking=True,
    )

    status = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This Fomula 1 race status measn the driver finished the race without any disqualification and damage, and with a gap of less than 2 Laps compared with the race winner.",
        df=status,
        depend_on=["status"],
        thinking=True,
    )

    merged_df = pd.merge(results, drivers, on="driverId")
    merged_df = pd.merge(merged_df, status, on="statusId")

    merged_df = merged_df[["driverId"]].drop_duplicates()
    prediction = len(merged_df)

    return prediction, op.total_tokens


# formula_1
# select + order
def pipeline_8(args):
    query = "Rank circuits locates in East Asia and rank them based on the population of the location country in descending order. Answer with circuitId."
    answer = [17, 16, 22, 28, 35]
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    circuit = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The country locates in east asia.",
        df=circuits,
        depend_on=["country"],
        thinking=True,
    )
    print(circuit.shape[0])
    op2 = LogicalOrder(operand_type=OperandType.ROW)
    k = len(circuit)
    result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="population of the location country of this circuit.",
        depend_on="country",
        ascending=False,
        k=k,
        df=circuit,
        thinking=True,
    )
    prediction = result["circuitId"].tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# formula_1
# select + order
def pipeline_9(args):
    query = "Among constructors Lewis Hamilton has been served or is serving for, which one won the most podiums in Formula 1 Grand Prix in the 20th century? Answer with constructor name."
    answer = ["Ferrari"]
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    constructors = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Lewis Hamilton has served or is serving for this constructor.",
        df=constructors,
        depend_on=["name"],
        thinking=True,
    )
    print(constructors)

    op2 = LogicalOrder(operand_type=OperandType.ROW)
    result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="number of podiums won in Formula 1 Grand Prix in the 20th century",
        depend_on="name",
        ascending=False,
        k=1,
        df=constructors,
        thinking=True,
    )
    prediction = result["name"].tolist()

    return prediction, op1.total_tokens + op2.total_tokens


# european_football_2
# select + select
def pipeline_10(args):
    query = "Among Top 10 heaviest player, list players higher than basketball player Michael Jordan and has a Belgium originated name."
    answer = ["Kristof van Hout"]
    player = pd.read_csv("./databases/european_football_2/Player.csv")

    player = player.dropna(subset=["weight"])
    player = player.sort_values(by="weight", ascending=False)[:10]

    op = LogicalSelect(operand_type=OperandType.ROW)
    player = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This player is higher than Michael Jordan.",
        df=player,
        depend_on=["height"],
        thinking=True,
    )
    player = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This player comes from Belgium.",
        df=player,
        depend_on=["player_name"],
        thinking=True,
    )

    prediction = player["player_name"].tolist()
    return prediction, op.total_tokens


# european_football_2
# select + order
def pipeline_11(args):
    query = "Identify the leagues whose country is adjacent to France and is not an inland country. List the league with the longest name among them."
    answer = "Belgium Jupiler League"
    league = pd.read_csv("./databases/european_football_2/League.csv")
    country = pd.read_csv("./databases/european_football_2/Country.csv")

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    country = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This country is adjacent to France.",
        df=country,
        depend_on=["name"],
        thinking=True,
    )
    print(country)
    country = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This country is not an inland country.",
        df=country,
        depend_on=["name"],
        thinking=True,
    )
    country = country[["id"]]
    merged_df = pd.merge(country, league, left_on="id", right_on="country_id")
    op2 = LogicalOrder(operand_type=OperandType.ROW)
    merged_df = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="the length of the name",
        df=merged_df,
        depend_on="name",
        k=1,
        ascending=False,
        thinking=True,
    )

    return merged_df["name"][0], op1.total_tokens + op2.total_tokens


# codebase_community
# select + select
def pipeline_12(args):
    query = "Find all users in the codebase_community database whose display name starts with 'm' (case-insensitive), live in 'Vienna, Austria', and have a display name that sounds like a real human name. Return their display names."
    answer = ["Maciej Pasternacki", "Marcus Jones"]

    users = pd.read_csv("./databases/codebase_community/users.csv")
    users = users.dropna(subset=["DisplayName"])
    users["DisplayName"] = users["DisplayName"].astype(str)

    m_starters_mask = users["DisplayName"].str.lower().str.startswith("m")
    filtered_users = users[m_starters_mask]

    op = LogicalSelect(operand_type=OperandType.ROW)
    filtered_users = filtered_users[["Location", "DisplayName"]]
    filtered_users = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This user lives in Vienna, Austria.",
        df=filtered_users,
        depend_on=["Location"],
        thinking=True,
    )
    print(filtered_users)

    filtered_users = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This DisplayName sounds like a real human name.",
        df=filtered_users,
        depend_on=["DisplayName"],
        thinking=True,
    )
    prediction = filtered_users["DisplayName"].tolist()
    return prediction, op.total_tokens


# codebase_community
# select + impute
def pipeline_13(args):
    query = "Among posts with top 5 highest-Viewcount, filter those that contain external links in their post body, and extract all external links from these posts."
    answer = ['http://onlinestatbook.com/2/analysis_of_variance/one-way.html']
    posts = pd.read_csv("./databases/codebase_community/posts.csv")
    posts = posts.sort_values(by="ViewCount", ascending=False)[:5]

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_posts = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This post body contains external link.",
        df=posts,
        depend_on=["Body"],
        thinking=True,
    )
    print(len(filtered_posts))
    print(filtered_posts['Body'])

    op2 = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Extract the external links in the post body.",
        df=filtered_posts,
        depend_on="Body",
        new_col="links",
        thinking=True,
    )
    prediction = result["links"].tolist()

    return prediction, op1.total_tokens + op2.total_tokens


# santos
# selet + match
def pipeline_14(args):
    query = "Among state agencies in the ‘state_expenditures’ table with total payments exceeding 200,000 USD but less than 200,000 GBP (calculated using the peak exchange rate in 2015), list the agencies whose functions are similar to those of Canada's departments responsible for 'Agriculture and Agri-Food' as found in the ‘rm-mr-2009-eng’ table."
    answer = "[['AGRICULTURE', 'Agriculture and Agri-Food Department']]"
    state_expenditures = pd.read_csv("./databases/santos/state_expenditures.csv")
    rm_2009 = pd.read_csv("./databases/santos/rm-mr-2009-eng.csv")

    state_expenditures = state_expenditures[state_expenditures['Payments Total'] > 200000]
    print(len(state_expenditures))
    print(state_expenditures)
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_administrations = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This cost in (USD) is less than 200000 GBP, using the highest exchange rate in 2015.",
        df=state_expenditures,
        depend_on=["Payments Total"],
        thinking=True,
    )

    filtered_administrations = filtered_administrations[['Agency Name']].drop_duplicates()
    print(filtered_administrations)

    rm_2009 = rm_2009[rm_2009['MINE']=='Agriculture and Agri-Food']
    rm_2009 = rm_2009[['DEPT_EN_DESC']]

    print(rm_2009)

    op2 = LogicalMatch(OperandType.CELL)
    pred = op2.execute(impl_type=ImplType.LLM_ALL, 
                           condition="The agency and the DEPT have similar function.", 
                           left_df = filtered_administrations, 
                           right_df = rm_2009, 
                           left_on = 'Agency Name',
                           right_on = 'DEPT_EN_DESC',
                           thinking=True)
    prediction = pred.values.tolist()
    
    return prediction, op1.total_tokens + op2.total_tokens

# santos
# select + match
def pipeline_15(args):
    query = "Among the records in the ‘dft-monthly-spend-201005’ table where the British Transport Police’s expenses on 28/05/2010 are greater than 10,000 USD, count how many have matching entries for software coding expenses on the same day in the transparency report from the ‘V3.1_-_Final__csv__May_2013__25k_Transprncy_rpt’ table."
    answer = 16
    dft_monthly_spend = pd.read_csv("./databases/santos/dft-monthly-spend-201005.csv")
    transprncy_rpt = pd.read_csv(
        "./databases/santos/V3.1_-_Final__csv__May_2013__25k_Transprncy_rpt.csv"
    )
    dft_monthly_spend = dft_monthly_spend[
        dft_monthly_spend["Entity"] == "British Transport Police"
    ]
    dft_monthly_spend = dft_monthly_spend[dft_monthly_spend["Date"] == "28/05/2010"]
    transprncy_rpt = transprncy_rpt[transprncy_rpt["Expense type"] == "Software coding"]
    transprncy_rpt = transprncy_rpt[["Date"]]
    dft_monthly_spend = dft_monthly_spend[["Date", "Amount"]]
    print(dft_monthly_spend)

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    merged_df = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The amount expense in GBP is higher than 10000 USD.",
        df=dft_monthly_spend,
        depend_on=["Amount"],
        thinking=True,
    )

    print(len(merged_df), len(transprncy_rpt))

    op2 = LogicalMatch(OperandType.CELL)
    merged_df = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The two date shares the same day of different time.",
        left_df=merged_df,
        right_df=transprncy_rpt,
        left_on="Date",
        right_on="Date",
        thinking=True,
    )

    merged_df = merged_df.drop_duplicates()
    prediction = merged_df.shape[0]
    return prediction, op1.total_tokens + op2.total_tokens


# santos
# match + impute
def pipeline_16(args):
    query = "List the officials in ‘travel-exp-April-June-2018’ table who share the same first name with senior officials from the ‘home_office_senior_officials_travel_data_return’ table, along with their destinations and the continents where these destinations are located."
    answer = [
        ["Mark Bryson-Richardson", "Paris, France", "Europe"],
        ["Richard Montgomery", "Kathmandu/ Yangon/Manilla", "Asia"],
    ]

    home_office_travel = pd.read_csv(
        "./databases/santos/home_office_senior_officials_travel_data_return.csv"
    )
    travel_exp_2018 = pd.read_csv("./databases/santos/travel-exp-April-June-2018.csv")

    home_office_officials = home_office_travel[["Name of Official"]].drop_duplicates()
    travel_2018_officials = travel_exp_2018[
        ["Senior Officials Name", "Destination"]
    ].drop_duplicates(subset=["Senior Officials Name"])

    print(len(home_office_officials), len(travel_2018_officials))
    print(home_office_officials)
    print(travel_2018_officials)

    op1 = LogicalMatch(OperandType.CELL)
    merged_df = op1.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The two senior government official names shares the same first name.",
        left_df=home_office_officials,
        right_df=travel_2018_officials,
        left_on="Name of Official",
        right_on="Senior Officials Name",
        thinking=True,
    )
    print(merged_df.columns)
    merged_df = merged_df[["right_Senior Officials Name", "right_Destination"]]

    op2 = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Impute the continent where this destination locates.",
        df=merged_df,
        depend_on="right_Destination",
        new_col="continent",
        thinking=True,
    )

    prediction = result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# santos
# select + match
def pipeline_17(args):
    query = "From the ‘pca-human-wildlife-coexistence-animals-involved-detailed-records’ table, identify the 50 most recent wildlife incidents involving deers. For each incident, list the protected heritage area where it occurred, if that area is in a country included in the ‘cihr_co-applicant_200607’ table."
    answer = "[['Elk Island National Park of Canada', 'Canada'], ['Gros Morne National Park of Canada', 'Canada']]"

    cihr = pd.read_csv("./databases/santos/cihr_co-applicant_200607.csv")
    wildlife = pd.read_csv("./databases/santos/pca-human-wildlife-coexistence-animals-involved-detailed-records.csv")

    wildlife = wildlife.sort_values(by='Incident Number', ascending=False)[:50]
    print(wildlife['Species Common Name'])
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    beers = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The species is a deer.",
        df=wildlife,
        depend_on=["Species Common Name"],
        thinking=True,
    )

    beers = beers[['Protected Heritage Area']].drop_duplicates()
    cihr= cihr[['CountryEN']].drop_duplicates()

    print(beers)
    print(cihr)

    op2 = LogicalMatch(OperandType.CELL)
    merged_df = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The protected heritage area location is in the country.", 
        left_df=beers, 
        right_df=cihr, 
        left_on='Protected Heritage Area',
        right_on='CountryEN',
        thinking=True
    )
    prediction = merged_df[['left_Protected Heritage Area', 'right_CountryEN']].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# santos
# match + order
def pipeline_18(args):
    query = "List the matching addresses that represent the same street location in both the ‘pubs’ and ‘community_centres’ tables within the Barking and Dagenham borough, sorted by the latitude of the street location."
    answer = [["Billet Road / Rose Lane", "Rose Lane"], ["109 Bastable Avenue / Charlton Crescent", "Bastable Avenue"]]

    pubs_data = pd.read_csv("./databases/santos/pubs.csv")
    community_centres_data = pd.read_csv("./databases/santos/community_centres.csv")

    # Filter to get unique addresses from both datasets
    pubs_data = pubs_data[pubs_data["borough_name"] == "Barking and Dagenham"]
    community_centres_data = community_centres_data[
        community_centres_data["borough_name"] == "Barking and Dagenham"
    ]
    pubs_addresses = pubs_data[["address1"]].drop_duplicates()
    community_addresses = community_centres_data[["address1"]].drop_duplicates()

    print(len(pubs_addresses), len(community_addresses))
    print(pubs_addresses)
    print(community_addresses)

    op1 = LogicalMatch(OperandType.CELL)
    merged_df = op1.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The two addresses represent the same street location.",
        left_df=pubs_addresses,
        right_df=community_addresses,
        left_on="address1",
        right_on="address1",
        thinking=True,
    )
    merged_df = merged_df.drop_duplicates()

    op2 = LogicalOrder(operand_type=OperandType.ROW)
    k = len(merged_df)
    merged_df = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="the Latitude of street.",
        depend_on="right_address1",
        ascending=True,
        k=k,
        df=merged_df,
        thinking=True,
    )

    prediction = merged_df.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# santos
# match + select
def pipeline_19(args):
    query = "List suppliers that appear in both the '01.Apr_2018' table and the '2015_05_expenditure' table that represent the same entity, and offer professional medical supplies."
    answer = "[['Nhs Supply Chain', 'NHS SUPPLY CHAIN'], ['Novartis Pharmaceuticals Uk Ltd', 'NOVARTIS PHARMACEUTICALS UK LTD'], ['Philips Healthcare', 'PHILIPS HEALTHCARE'], ['Nhs Blood And Transplant', 'NHS BLOOD & TRANSPLANT']]"
    left = pd.read_csv("./databases/santos/01.Apr_2018.csv")
    right = pd.read_csv("./databases/santos/2015_05_expenditure.csv")
    left = left[['Supplier']].drop_duplicates()
    right = right[['Supplier']].drop_duplicates()
    print(len(left), len(right))

    op1 = LogicalMatch(OperandType.CELL)
    merged_df = op1.execute(impl_type=ImplType.LLM_ALL, 
                           condition="The two suppliers are the same supplier.", 
                           left_df=left, 
                           right_df=right, 
                           left_on='Supplier',
                           right_on='Supplier',
                           thinking=True)

    op2 = LogicalSelect(operand_type=OperandType.ROW)
    merged_df = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The company is related to professional medical supplies.",
        df=merged_df,
        depend_on="left_Supplier",
        thinking=True,
    )
    prediction = merged_df.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# mondial_geo
# select + match
def pipeline_20(args):
    query = "For each Asian city with more than 8 million people, what is the closest sea?"
    answer = "[['Jakarta', 'Sunda Sea'], ['Karachi', 'Arabian Sea'], ['Mumbai', 'Arabian Sea'], ['Seoul', 'Yellow Sea']]"

    city = pd.read_csv("./databases/mondial_geo/city.csv")
    city = city[city['Population'] > 8000000]
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    print(len(city))
    print(city)
    
    op1 = LogicalSelect(OperandType.ROW)
    city = op1.execute(
        impl_type= ImplType.LLM_ONE,
        condition="The city locates in an Asian Country.",
        df=city,
        depend_on=["Name","Country","Province"],
        thinking = True
    )
    print(city)
    print(len(city))
    print(len(sea))
    print(sea)

    op2 = LogicalMatch(OperandType.CELL)
    merged_df = op2.execute(impl_type=ImplType.LLM_ALL,
                    condition="Among all provided seas, map each city to its closest sea.", 
                    left_df=city, 
                    right_df=sea, 
                    left_on='Name',
                    right_on='Name',
                    thinking=True)

    prediction = merged_df[['left_Name', 'right_Name']].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# mondial_geo
# order + match
def pipeline_21(args):
    query = "List the continents that adjoin each of the world's three smallest seas."
    answer = "[['Kattegat', 'Europe'], ['Skagerrak', 'Europe'], ['Sea of Azov', 'Europe']]"
   
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    continent = pd.read_csv("./databases/mondial_geo/continent.csv")

    print(sea)
    op1 = LogicalOrder(operand_type=OperandType.ROW)
    sea = op1.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The area of the sea.",
        df=sea,
        depend_on="Name", 
        k=3,
        ascending=True,
        thinking = True
    )

    print(sea)
    print(continent)

    # 36*5
    op2 = LogicalMatch(OperandType.CELL)
    merged_df = op2.execute(impl_type=ImplType.LLM_ALL,
                           condition="The sea adjoins the continent.", 
                           left_df=sea, 
                           right_df=continent, 
                           left_on='Name',
                           right_on='Name',
                           thinking=True)
    prediction = merged_df[['left_Name', 'right_Name']].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# disney
# select + match
def pipeline_22(args):
    query = "Calculate the average total revenue for years in which musical movies were released after the Beijing Summer Olympics, by matching movie release dates with revenue data from the same year."
    answer = [[48813.0]]
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")
    revenue = pd.read_csv("./databases/disney/revenue.csv")

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_revenue = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This year is later than the year of Beijing Summer Olympics.",
        df=revenue,
        depend_on=["Year"],
        thinking=True,
    )

    print(filtered_revenue)

    movies_total_gross = movies_total_gross[movies_total_gross["genre"] == "Musical"]
    print(len(filtered_revenue), len(movies_total_gross))

    op2 = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The two date are in the same year.",
        left_df=movies_total_gross,
        right_df=filtered_revenue,
        left_on="release_date",
        right_on="Year",
        thinking=True,
    )

    prediction = merged_df.agg(avgTotalRevenue=("right_Total", "mean"))

    return prediction.values.tolist(), op1.total_tokens + op2.total_tokens


# disney
# select + impute
def pipeline_23(args):
    query = "Calculate the average profit margin (total gross divided by inflation-adjusted gross) for musical movies that were released after the end of World War II."
    answer = 0.5299
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")

    # Filter movies to only include comedy genre
    movies_total_gross = movies_total_gross[movies_total_gross["genre"] == "Musical"]
    print(len(movies_total_gross))
    print(movies_total_gross)

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    movies_total_gross = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This date is after end of World War II.",
        df=movies_total_gross,
        depend_on=["release_date"],
        thinking=True,
    )

    # Calculate profit margin using LogicalImpute
    op2 = LogicalImpute(operand_type=OperandType.COLUMN)
    movies_total_gross = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the profit margin as total gross divided by inflation-adjusted gross. Answer with a float value.",
        df=movies_total_gross,
        depend_on=["total_gross", "inflation_adjusted_gross"],
        new_col="profit_margin",
        thinking=True,
    )

    # Calculate average profit margin (convert to int first)
    prediction = movies_total_gross["profit_margin"].astype(float).mean().round(4)
    print(prediction)

    return prediction, op1.total_tokens + op2.total_tokens


# disney
# select + impute + order
def pipeline_24(args):
    query = "List the directors of adventure movies with an MPAA rating of 'PG-13', released after the date of Barack Obama's second inauguration. Return the list of directors, sorted in ascending order by their birthdays."
    answer = [['Alan Taylor'], ['J. J. Abrams'], ['Scott Derrickson'], ['James Gunn'], ['Gareth Edwards']]
    director = pd.read_csv("./databases/disney/director.csv")
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")

    # Filter characters from movies released in the 1990s
    movies_total_gross = movies_total_gross[movies_total_gross["genre"] == "Adventure"]
    movies_total_gross = movies_total_gross[movies_total_gross["MPAA_rating"] =='PG-13']
    
    print(movies_total_gross)

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    movies_total_gross = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This movie was released after the inauguration day of Barack Obama's second presidency.",
        df=movies_total_gross,
        depend_on=["release_date"],
        thinking=True,
    )
    
    op2 = LogicalImpute(operand_type=OperandType.COLUMN)
    movies_total_gross = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Director of this movie.",
        df=movies_total_gross,
        depend_on="movie_title",
        new_col="director",
        thinking=True,
    )

    print(movies_total_gross)

    op3 = LogicalOrder(operand_type=OperandType.ROW)
    k = len(movies_total_gross)
    result = op3.execute(
        impl_type=ImplType.LLM_ALL,
        condition="birthday of the director of this movie.",
        depend_on=["movie_title", "director"],
        ascending=True,
        k=k,
        df= movies_total_gross,
        thinking=True,
    )

    prediction = result[['director']].values.tolist()

    return prediction, op1.total_tokens + op2.total_tokens + op3.total_tokens


# car_retails
# select, groupby
def pipeline_25(args):
    query = "For all shipped orders with non-empty comments that requested DHL shipping, cluster the customer locations into two corresponding buckets by urbanization level. For each cluster label, count how many customers fall into that cluster."
    answer = [['Rural', 1], ['Urban', 5]]
    order = pd.read_csv("./databases/car_retails/orders.csv")
    customer = pd.read_csv("./databases/car_retails/customers.csv")

    order = order[order["status"] == "Shipped"].dropna(subset=("comments"))
    op1 = LogicalSelect(operand_type=OperandType.ROW) 
    order = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This commnet asked for DHL shipping.",
        df=order,
        depend_on=["comments"],
        thinking=True,
    )
    merged_df = pd.merge(order, customer, on="customerNumber")

    print(merged_df[['addressLine1', 'addressLine2', 'city', 'state', 'postalCode', 'country']])

    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Cluster these locations into two buckets by urbanization level.",
        df=merged_df,
        depend_on=[
            "addressLine1",
            "addressLine2",
            "city",
            "state",
            "postalCode",
            "country",
        ],
        thinking=True,
    )

    grouped_result = grouped_result[['customerNumber','cluster_name']].drop_duplicates()

    grouped_result = (
        grouped_result.groupby("cluster_name")
        .agg(avgTotalRevenue=("customerNumber", "count"))
        .reset_index()
    )

    prediction = grouped_result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# mondial_geo
# select + groupby
def pipeline_26(args):
    query = "Among European countries with a population greater than one-thirtieth of China's population, group them by level of economic development and count the number of countries in each group."
    answer = [["Developed", 4], ["Developing", 1], ["Emerging", 2]]

    country = pd.read_csv("./databases/mondial_geo/country.csv")
    encompasses = pd.read_csv("./databases/mondial_geo/encompasses.csv")

    european_countries = encompasses[encompasses["Continent"] == "Europe"][
        "Country"
    ].tolist()
    european_data = country[country["Code"].isin(european_countries)]

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_countries = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This country has population over one-thirtieth of China's population.",
        df=european_data,
        depend_on=["Population"],
        thinking=True,
    )
    print(filtered_countries)

    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Group these countries by their economic development level into categories like 'Developed', 'Developing', and 'Emerging'.",
        df=filtered_countries,
        depend_on=["Name", "Population", "Area"],
        thinking=True,
    )

    result = (
        grouped_result.groupby("cluster_name")
        .agg(country_count=("Name", "count"))
        .reset_index()
    )

    prediction = result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# music
# select + groupby
def pipeline_27(args):
    query = "For Eminem's songs released after 2000 in an edited version album, group them by decade and count the number of songs in each decade."
    answer = [["2000s", 15], ["2010s", 19]]  

    amazon_music = pd.read_csv("./databases/music/amazon_music.csv")

    albums = amazon_music[amazon_music["Artist_Name"] == "Eminem"]
    albums["Released"] = pd.to_datetime(albums["Released"])
    filtered_albums = albums[albums["Released"].dt.year > 2000]

    print(len(albums))
    print(albums[['Album_Name', 'Song_Name']])
    
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_albums = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This album is an edited version album.",
        df=filtered_albums,
        depend_on=["Album_Name"],
        thinking=True,
    )
    print(len(filtered_albums))
    
    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_albums = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Group these release date by decade (2000s, 2010s, 2020s).",
        df=filtered_albums,
        depend_on=["Released"],
        thinking=True,
    )

    result = (
        grouped_albums.groupby("cluster_name")
        .agg(album_count=("Song_Name", "count"))
        .reset_index()
    )

    prediction = result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# formula_1
# select + match
def pipeline_28(args):
    query = "Which circuits located in Europe are named after a constructor originating from Europe? Answer with the circuit name."
    answer = ['Autodromo Enzo e Dino Ferrari', 'Red Bull Ring']
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")
    
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    circuits = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The country is in Europe",
        df=circuits,
        depend_on="country",
        thinking = True,
    )
    print(circuits)
    print(circuits.shape[0])  # 37
    constructors = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The nationality corresponds to an European country",
        df=constructors,
        depend_on="nationality",
        thinking = True,
    )
    print(constructors)
    print(constructors.shape[0])  # 150
    
    op2 = LogicalMatch(operand_type=OperandType.CELL)
    merged = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The circuit name is named after the provided constructor name",
        left_df=circuits,
        right_df=constructors,
        left_on="name",
        right_on="name",
        thinking = True,
    )
    print(merged)
    return merged["left_name"].tolist(), op1.total_tokens + op2.total_tokens


# beer_factory
# select + cluster
def pipeline_29(args):
    query = "Among US brands that receive positive comments, cluster them by their locations into 'US-East', 'US-West', and 'US-Central', based on the geographical location of their origin with respect to the United States."
    answer = [["Captain Eli's", 'US-East'], ["Fitz's", 'US-Central'], ['Sprecher', 'US-Central'], ['Bulldog', 'US-West'], ["Sparky's Fresh Draft", 'US-West'], ['Gales', 'US-Central']]
    beer_brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    beer_review = pd.read_csv("./databases/beer_factory/rootbeerreview.csv")

    beer_review = beer_review.dropna(subset=["Review"])
    print(beer_review['Review'])
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_review = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This comment is positive.",
        df=beer_review,
        depend_on=["Review"],
        thinking=True,
    )
    print(filtered_review)
    print(len(filtered_review))

    merged_df = pd.merge(filtered_review, beer_brand, on="BrandID")
    
    merged_df = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This state is in the United States.",
        df=merged_df,
        depend_on=["State"],
        thinking=True,
    )

    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_brand = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Cluster by the US state into 'US-East', 'US-West' and 'US-Central' based on geographical location.",
        df=merged_df,
        depend_on=["State"],
        thinking=True,
    )

    print(grouped_brand.columns)
    prediction = grouped_brand[["BrandName", "cluster_name"]].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens



# beer_factory
# select + cluster
def pipeline_30(args):
    query = "Among root beer brands that began their first brew more than 50 years before the Beijing Summer Olympics, cluster them by their location's Köppen–Geiger climate types (Letter Code), then count how many brands belong to each climate cluster."
    answer = [['Cfa', 3], ['Csa', 1], ['Csb', 2], ['Dfa', 5]]
    beer_brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    print(beer_brand)
    
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_brand = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This year is more than 50 years before the Beijing Summer Olympics.",
        df=beer_brand,
        depend_on=["FirstBrewedYear"],
        thinking=True,
    )

    # print(len(filtered_brand))
    print(filtered_brand[['BrandName', 'City', 'State', 'Country']])

    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_brands = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Cluster these locations by their Köppen–Geiger climate types(ONLY with letter codes).",
        df=filtered_brand,
        depend_on=["City", "State", "Country"],
        thinking=True,
    )
    
    result = (
        grouped_brands.groupby("cluster_name")
        .agg(brand_count=("BrandName", "count"))
        .reset_index()
    )
    
    prediction = result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# beer_factory
# select + order
def pipeline_31(args):
    query = "Identify all male customers with an Outlook email address who reside in California's capital city. Of these, select the five customers living furthest north and provide their first and last names."
    answer =  [['Trevor', 'Matich'], ['Scott', 'Boras'], ['John', 'Bowker'], ['Thomas', 'Kelly'], ['Lance', 'Briggs']]
    customer = pd.read_csv("./databases/beer_factory/customers.csv")

    customer = customer[customer['Email'].str.endswith('@outlook.com')]
    customer = customer[customer['Gender'] == 'M']
    
    print(len(customer))
    print(customer.columns)
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_customer = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This City is the capital of California.",
        df=customer,
        depend_on=["City"],
        thinking=True,
    )

    print(len(filtered_customer))
    print(filtered_customer[['First', 'Last', 'StreetAddress', 'City', 'State']])
    op2 = LogicalOrder(operand_type=OperandType.ROW)
    k = len(filtered_customer)
    result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="the latitude of this location",
        depend_on=['StreetAddress','City','State'],
        ascending=False,
        k=5,
        df=filtered_customer,
        thinking = True,
    )

    print(result)
    prediction = result[['First','Last']].values.tolist()

    return prediction, op1.total_tokens + op2.total_tokens


# menu
# select + cluster
def pipeline_32(args):
    query = "Consider the spicy dishes that appear in menus from New York restaurants after 1970. Group these dishes by their countries of origin, and count how many dishes belong to each cluster."
    answer =[['Germany', 1], ['India', 2], ['USA', 3]] 

    menu = pd.read_csv("./databases/menu/Menu.csv")
    dish = pd.read_csv("./databases/menu/Dish.csv")
    menu_item = pd.read_csv("./databases/menu/MenuItem.csv")
    menu_page = pd.read_csv("./databases/menu/MenuPage.csv")

    menu = menu.rename(columns = {"id":"menu_id"})
    menu_page = menu_page.rename(columns = {"id":"menu_page_id"})
    dish = dish.rename(columns = {"id":"dish_id", "name": "dish_name"})

    # Filter menus from New York after 1950
    menu["date"] = pd.to_datetime(menu["date"], errors="coerce")
    ny_menus = menu[
        (menu["place"].str.contains("NEW YORK", case=False, na=False)) & 
        (menu["date"].dt.year > 1970)
    ]
    print(ny_menus)


    # Join tables to get dishes from these menus
    merged_df = pd.merge(ny_menus, menu_page, on = "menu_id")
    merged_df = pd.merge(merged_df, menu_item, on="menu_page_id")
    merged_df = pd.merge(merged_df, dish, on="dish_id")

    merged_df = merged_df[['dish_name', 'description']].drop_duplicates()
    print(merged_df)
    
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    filtered_dishes = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This dish is spicy.",
        df=merged_df,
        depend_on=["dish_name"],
        thinking=True,
    )

    print(filtered_dishes[["dish_name"]])
    
    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_dishes = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Group these dishes by their country of origin.",
        df=filtered_dishes,
        depend_on=["dish_name"],
        thinking=True,
    )
    
    print(grouped_dishes[["dish_name"]])
    
    result = (
        grouped_dishes.groupby("cluster_name")
        .agg(dish_count=("dish_name", "count"))
        .reset_index()
    )
    
    prediction = result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# menu
# select + order
def pipeline_33(args):
    query = "Find all expensive dishes (lowest price > $480) that appeared in hotel menus, then sort them by the time they first appeared from earliest to latest. Answer with the dish names."
    answer = [['Helados diversos'], ['luso']]
    
    menu = pd.read_csv("./databases/menu/Menu.csv")
    dish = pd.read_csv("./databases/menu/Dish.csv")
    menu_item = pd.read_csv("./databases/menu/MenuItem.csv")
    menu_page = pd.read_csv("./databases/menu/MenuPage.csv")


    menu = menu.rename(columns = {"id":"menu_id"})
    menu_page = menu_page.rename(columns = {"id":"menu_page_id"})
    dish = dish.rename(columns = {"id":"dish_id", "name": "dish_name"})

    expensive_dishes = dish[dish["lowest_price"] > 480.0]
    expensive_dishes = expensive_dishes.drop_duplicates(subset=["dish_name"])

    merged_df = pd.merge(menu, menu_page, on = "menu_id")
    merged_df = pd.merge(merged_df, menu_item, on="menu_page_id")
    merged_df = pd.merge(merged_df, expensive_dishes, on="dish_id")
    
    print(len(merged_df))
    
    print(merged_df)

    # Filter hotel menus
    op1= LogicalSelect(operand_type=OperandType.ROW)
    hotel_menus = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This menu is from a hotel restaurant.",
        df=merged_df,
        depend_on=["venue", "sponsor", "place"],
        thinking=True,
    )
    print(len(hotel_menus))

    # Sort by first appearance date
    op2 = LogicalOrder(operand_type=OperandType.ROW)
    k = len(hotel_menus)
    result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="the first appearance time of this dish",
        depend_on=["first_appeared"],
        ascending=True,
        k=k,
        df=hotel_menus,
        thinking=True,
    )
    
    prediction = result[["dish_name"]].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# menu
# select + cluster
def pipeline_34(args):
    query = "Consider dishes that contain seafood ingredients and have a lowest price greater than 100 USD. Cluster them by their preparation methods into 'Baked', 'Grilled', 'Fried', 'Raw', 'Sautéed', 'Steamed' and 'Others'. Count how many dishes use each preparation method."
    answer = [['Baked', 4], ['Fried', 2], ['Grilled', 2], ['Others', 10], ['Raw', 3], ['Sautéed', 3], ['Steamed', 4]]
    dish = pd.read_csv("./databases/menu/Dish.csv")

    expensive_dishes = dish[dish["lowest_price"] > 100.0]
    expensive_dishes = expensive_dishes.drop_duplicates(subset=["name"])

    # Filter seafood dishes
    op1 = LogicalSelect(operand_type=OperandType.ROW)   
    seafood_dishes = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This dish contains seafood ingredients.",
        df=expensive_dishes,
        depend_on=["name"],
        thinking=True,
    )

    print(len(seafood_dishes))
    
    # Group by preparation method
    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_dishes = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Cluster these seafood dishes by their preparation methods into 'Baked', 'Grilled', 'Fried', 'Raw', 'Sautéed', 'Steamed' and 'Others'.",
        df=seafood_dishes,
        depend_on=["name"],
        thinking=True,
    )
    
    result = (
        grouped_dishes.groupby("cluster_name")
        .agg(dish_count=("name", "count"))
        .reset_index()
    )
    
    prediction = result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# olympics
# select + cluster
def pipeline_35(args):
    query = "Filter Asian cities that have held the Olympic Games, group them by country, and count how many times each country has hosted the Olympic Games."
    answer = [["China", 1], ["Japan", 3], ["South Korea", 1]] 

    city = pd.read_csv("./databases/olympics/city.csv")
    game = pd.read_csv("./databases/olympics/games.csv")
    games_city = pd.read_csv("./databases/olympics/games_city.csv")

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    south_america_hosts = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This host city is located in Asia.",
        df=city,
        depend_on=["city_name"],
        thinking=True,
    )

    merged_df = pd.merge(south_america_hosts, games_city, left_on='id', right_on='city_id')
    merged_df = pd.merge(merged_df, game, left_on='games_id', right_on='id')

    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_hosts = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Group these Asian host cities by their country.",
        df=merged_df,
        depend_on=["city_name"],
        thinking=True,
    )
    print(grouped_hosts.columns)
    result = (
        grouped_hosts.groupby("cluster_name")
        .agg(country_count=("games_name", "count"))
        .reset_index()
    )

    prediction = result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# formula_1 
# select + select + order
def pipeline_36(args):
    query = "List all drivers who were born in Europe, participated in the 2017 Monaco Grand Prix, were born after January 1, 1991, and are younger than Alex Albon. Rank them by their debut year, from earliest to latest."
    answer = [['Max', 'Verstappen'], ['Esteban', 'Ocon']]

    drivers = pd.read_csv("./databases/formula_1/drivers.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")

    races  =races[(races['year'] == 2017) & (races['name'] == 'Monaco Grand Prix')]

    drivers['dob'] = pd.to_datetime(drivers['dob'])
    drivers = drivers[drivers['dob'] > '1991-01-01']   
    drivers['dob'] = drivers['dob'].dt.strftime('%Y-%m-%d')
    print(drivers)

    merged_df = pd.merge(races, results, on = "raceId")
    merged_df = pd.merge(merged_df, drivers, on = "driverId")
    print(merged_df)

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    merged_df = op1.execute(impl_type=ImplType.LLM_ONE,
                        condition = "The provided birth date is later than the birth date of Alex Albon.",
                        df = merged_df, 
                        depend_on = ['forename', 'surname', 'dob'],
                        thinking = True)

    print(merged_df)

    merged_df = op1.execute(impl_type=ImplType.LLM_ONE,
                        condition = "The nationality corresponds to a country in Europe.",
                        df = merged_df, 
                        depend_on = ['nationality'],
                        thinking = True)

    print(merged_df)

    op2 = LogicalOrder(operand_type=OperandType.ROW)
    k = len(drivers)
    result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The debut year of this formula 1 driver.",
        depend_on=['forename', 'surname'],
        ascending=True,
        k=k,
        df=merged_df,
        thinking = True
    )
    
    print(result)
    prediction = result[['forename', 'surname']].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# formula_1
# select + impute
def pipeline_37(args):
    query = "List drivers who were born after 1980, have won a world championship, and have not competed in either GP2 or Formula 2."
    answer = "[['Fernando', 'Alonso'], ['Sebastian', 'Vettel'], ['Max', 'Verstappen']]"

    drivers = pd.read_csv("./databases/formula_1/drivers.csv").dropna()
    drivers = drivers[pd.to_datetime(drivers['dob']).dt.year > 1980]
    drivers = drivers[['forename', 'surname']]
    print(drivers)
    
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    result = op1.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This driver is a Formula 1 world champion.",
                        df = drivers, 
                        depend_on = ['forename', 'surname'],
                        thinking = True)

    op2 = LogicalSelect(operand_type=OperandType.ROW)
    result = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition = "This driver has not been raced in GP2 or Formula 2.",
        df = result, 
        depend_on = ['forename', 'surname'],
        thinking = True
    )
    
    prediction = result.values.tolist();
    return prediction, op1.total_tokens + op2.total_tokens


# electronics
# select + select
def pipeline_38(args):
    query = "In the Amazon table, find the IDs for Sony laptop chargers that originally cost more than £20, using the peak 2020 exchange rate."
    answer = [[758], [2543], [3790]]
    products = pd.read_csv("./databases/electronics/amazon.csv")
    products = products[products['Brand'] == 'Sony']

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This product is a laptop charger.",
                        df = products, 
                        depend_on = ['Name'],
                        thinking = True)

    result = result.dropna(subset=['Original_Price'])
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This price in USD is larger than 20 GBP using the peak exchange rate of year 2020.",
                        df = result, 
                        depend_on = ['Original_Price'],
                        thinking = True)
    
    prediction = result[['ID']].values.tolist()
    return prediction, op.total_tokens

# books
# select + select
def pipeline_39(args):
    query = "Filter ids of books about natural science published between Jan 2000 and Feb 2000, and written in original British English."
    answer = ['8036', '9538']

    books = pd.read_csv("./databases/books/book.csv")
    language = pd.read_csv("./databases/books/book_language.csv")

    books['publication_date'] = pd.to_datetime(books['publication_date'])
    books = books[(books['publication_date'] >= '2000-01-01') & (books['publication_date'] <= '2000-02-29')]
    books['publication_date'] = books['publication_date'].dt.strftime('%Y-%m-%d')

    print(books)
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This book is about natural science.",
                        df = books, 
                        depend_on = ['title'],
                        thinking = True)
    print(result.columns)
    print(result)

    merged_df = pd.merge(result, language, on="language_id")
    
    print(merged_df)
    merged_df = op.execute(
            impl_type=ImplType.LLM_ONE,
            condition = "This language is the original British English.",
            df = merged_df, 
            depend_on = ['language_name'],
            thinking = True
    )
    print(merged_df.columns)
    print(merged_df)
    if merged_df.shape[0] > 0:
        prediction = merged_df[['book_id']].values.tolist()
    else:
        prediction = []
    return prediction, op.total_tokens


# shipping
# select + select
def pipeline_40(args):
    query = "How many customers received shipments in 2016 and after the 2016 Rio Summer Olympics from a state that is adjacent to their own home state?"
    answer = 14
    
    shipment = pd.read_csv("./databases/shipping/shipment.csv")
    city = pd.read_csv("./databases/shipping/city.csv")
    customer = pd.read_csv("./databases/shipping/customer.csv")
    
    shipment = shipment[(pd.to_datetime(shipment['ship_date']).dt.year == 2016)]
    # Filter records where ship_date > 2016 Aug 21
    # shipment['ship_date'] = pd.to_datetime(shipment['ship_date'], format='%Y-%m-%d')
    # shipment = shipment[shipment['ship_date'] > pd.to_datetime('2016-08-21')]


    print(len(shipment))
    op = LogicalSelect(operand_type=OperandType.ROW)
    shipment = op.execute(
                    impl_type=ImplType.LLM_ONE, 
                    condition="The date is after the end of Rio Summer Olympics.", 
                    df = shipment, 
                    depend_on = ['ship_date'],
                    thinking = True
                    )

    shipment = pd.merge(shipment, city, on='city_id')
    shipment = pd.merge(shipment, customer, on='cust_id')
    print(shipment.columns)
    
    print(shipment)
    
    df = op.execute(impl_type=ImplType.LLM_ONE, 
                    condition="The two states are geographically adjacent.", 
                    df = shipment, 
                    depend_on = ['state_x', 'state_y'],
                    thinking = True)

    prediction = df[['cust_name']].drop_duplicates().shape[0]
    print(prediction)
    return prediction, op.total_tokens

# car_retails
# select + impute
def pipeline_41(args):
    query = "Among orders that have been shipped and specified a preferred delivery company in the comments, how many days did it take to ship these orders on average?"
    answer = 3.3

    orders = pd.read_csv("./databases/car_retails/orders.csv")
    orders = orders[orders["status"] == "Shipped"]
    orders = orders.dropna(subset=["comments"])

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    orders = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This comment appoints or requests specific company for delivering.",
        df=orders,
        depend_on=["comments"],
        thinking=True,
    )
    print(orders)

    truth = ((pd.to_datetime(orders['shippedDate']) - pd.to_datetime(orders['orderDate'])).dt.days).tolist()
    print(len(truth))
    op2 = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op2.execute(impl_type=ImplType.LLM_ONE, 
                               condition="Calculate the number of days between shippedDate and orderDate", 
                               df = orders,
                               depend_on = ['shippedDate', 'orderDate'], 
                               new_col = 'days',
                               thinking = True)
    imputed_table['days'] = imputed_table['days'].map(int)
    pred = imputed_table['days'].tolist()

    avg = sum(pred) / len(pred) if pred else 0
    print(pred)

    return avg, op1.total_tokens + op2.total_tokens
    
# restaurants2
# select + select
def pipeline_42(args):
    query = "Which Yelp restaurants (votes > 3000) are located in Northeastern cities of the United States and serve cuisines that originate from Asia?"
    answer = [['Joeâs Shanghai'], ['Totto Ramen'], ['Ippudo NY']]
    
    yelp = pd.read_csv("./databases/restaurants2/yelp.csv")
    yelp = yelp[yelp['votes'] > 3000]
    print(yelp.shape[0]) # 22
    op = LogicalSelect(operand_type=OperandType.ROW)
    yelp = op.execute(impl_type=ImplType.LLM_ONE, 
                        condition="The zip belongs to a northeast city of USA", 
                        df = yelp, 
                        depend_on = ['zip'],
                        thinking = True)
    yelp = yelp[['name', 'cuisine']]
    print(yelp)
    
    yelp = op.execute(impl_type=ImplType.LLM_ONE, 
                        condition="The cuisine originates from Asia.", 
                        df = yelp, 
                        depend_on = ['cuisine'],
                        thinking = True)

    print(yelp)
    prediction = yelp[['name']].values.tolist()

    return prediction, op.total_tokens
    
# california_schools
# select + order 
def pipeline_43(args):
    query = "Filter schools with the top 30 FRPM counts that have school names that are more likely to be a human's full name, and rank them geographically from south to north by county. Answer with the school name."
    answer  = ['Hector G. Godinez', 'James A. Garfield Senior High', 'John H. Francis Polytechnic']
    
    schools = pd.read_csv("./databases/california_schools/schools.csv")
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")
    
    merged_df = pd.merge(schools, frpm, on='CDSCode')
    merged_df = merged_df.dropna(subset=['FRPM Count (K-12)'])
    merged_df = merged_df.sort_values(by='FRPM Count (K-12)', ascending=False)[:30]
    merged_df = merged_df[['County', 'School Name']]
    
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    merged_df = op1.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This schools sounds like a human's full name.",
                        df = merged_df, 
                        depend_on = ['School Name'],
                        thinking = True)

    print(merged_df)

    k = len(merged_df)
    op2 = LogicalOrder(operand_type=OperandType.ROW)
    result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="the latitude of this county",
        depend_on="County",
        ascending=True,
        k=k,
        df=merged_df,
        thinking = True
    )
                      
    prediction = result[['School Name']].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens


# mondial_geo
# select + order
def pipeline_44(args):
    query = "List lakes with a altitude larger than 500 and whose areas are larger than the area of Shanghai, and rank them based on their absolute distance to London (United Kingdom) in ascending order. Answer with the lake name."
    answer = "[['Lake Bangweulu'], ['Lake Tanganjika'], ['Lake Titicaca'], ['Lake Victoria'], ['Salar de Uyuni']]"
    
    lake = pd.read_csv("./databases/mondial_geo/lake.csv")
    lake = lake[lake['Altitude'] > 500]
    print(lake.shape) # 43
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    lake = op1.execute(impl_type = ImplType.LLM_ONE, 
                    condition = "The Area is larger than the area of Shanghai",
                    df = lake,
                    depend_on = ['Area'],
                    thinking = True)

    print(lake)

    k = len(lake)
    op2 = LogicalOrder(operand_type=OperandType.ROW)
    result = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="the absolute distance of this lake to London(United Kingdom).",
        depend_on="Name",
        ascending=True,
        k=k,
        df=lake,
        thinking = True
    )

    print(result)

    prediction = result[['Name']].values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens

# disney
# select + select + select
def pipeline_45(args):
    query = "List Disney movies released in the 21st Century that have a male villain and whose total gross is more than 95% of their inflation-adjusted gross. Answer with the movie name."
    answer = [['Frozen'], ['Big Hero 6']]

    characters = pd.read_csv("./databases/disney/characters.csv")
    director = pd.read_csv("./databases/disney/director.csv")
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")

    print(len(characters))
    op = LogicalSelect(OperandType.ROW)
    characters = op.execute(impl_type=ImplType.LLM_ONE, 
                            condition = "The Disney movie villian is male.",
                            df = characters, 
                            depend_on = ['villian'], 
                            thinking=True)

    print(len(characters))
    print(characters)

    characters = op.execute(impl_type=ImplType.LLM_ONE, 
                            condition = "The movie is released in 21th Century.",
                            df = characters, 
                            depend_on = ['release_date'], 
                            thinking=True)
    
    print(len(characters))
    print(characters)

    merged_df = pd.merge(director, characters, left_on="name", right_on="movie_title")
    merged_df = pd.merge(merged_df, movies_total_gross, on = "movie_title")

    print(merged_df)

    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="total_gross is larger than 95% of the inflation_adjusted_gross.",
        df=merged_df,
        depend_on=['total_gross', 'inflation_adjusted_gross'],
        thinking=True,
    )

    print(merged_df)
    prediction = merged_df[['movie_title']].values.tolist()
    return prediction, op.total_tokens
    

# formula_1
# select + select + cluster
def pipeline_46(args):
    query = "Group and count Formula 1 circuits by continent, but only include those built after the end year of the Vietnam War and that have hosted races for modern (21st-century) constructors."
    answer = [['Asia', 10], ['Europe', 7], ['North America', 6], ['Oceania', 1], ['South America', 1]]
    
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    constructorStandings = pd.read_csv("./databases/formula_1/constructorStandings.csv")

    print(constructors)
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    constructors = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This constructor have participated in at least one Grand Prix after 21st century.",
        df=constructors,
        depend_on=['constructorRef', 'name'],
        thinking=True,
    )

    print(constructors)
    print(circuits)

    circuits = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This circuit is build after the end year of Vietnam War.",
        df=circuits,
        depend_on=['circuitRef', 'name'],
        thinking=True,
    )

    print(len(circuits))

    circuits = circuits.rename(columns={"name": "circuit_name"})
    constructors = constructors.rename(columns={"name": "constructor_name"})
    merged_df = pd.merge(races, circuits, on="circuitId")
    merged_df = pd.merge(merged_df, constructorStandings, on="raceId")
    merged_df = pd.merge(merged_df, constructors, on="constructorId")

    merged_df = merged_df[['circuit_name', 'country']].drop_duplicates()
    print(merged_df)
    op2 = LogicalCluster(operand_type=OperandType.ROW)
    grouped_df = op2.execute(
                        impl_type=ImplType.LLM_ALL,
                        condition="Cluster the circuit by continent.", 
                        df = merged_df,
                        depend_on = ['circuit_name', 'country'],
                        thinking = True
    )

    print(grouped_df)
    print(grouped_df.columns)
    
    prediction = grouped_df.groupby('cluster_name').agg({'circuit_name': 'count'}).reset_index().values.tolist()
    
    return prediction, op1.total_tokens + op2.total_tokens


# student_club
# select + select + impute
def pipeline_47(args):
    query = "Calculate the average proportion of the remaining money to the budget amount for those monthly speaking events held before the day COVID-19 was declared a pandemic by the World Health Organization."
    answer = 0.535

    budget = pd.read_csv("./databases/student_club/budget.csv")
    event = pd.read_csv("./databases/student_club/event.csv")

    print(event['event_date'])
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    event = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This event happens before the declare day of COVID-19 by the World Health Organization.",
        df=event,
        depend_on=['event_date'],
        thinking=True,
    )

    print(event)
    event = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This event is a monthly speaking.",
        df=event,
        depend_on=['event_name'],
        thinking=True,
    )
    print(len(event))
    print(event)
    merged_df = pd.merge(event, budget, left_on="event_id", right_on="link_to_event")
    op2 = LogicalImpute(OperandType.COLUMN)
    merged_df = op2.execute(ImplType.LLM_ONE, 
                condition="impute the proportion of remaining money with respect to the amount.", 
                df = merged_df,
                depend_on = ['spent' ,'remaining', 'amount'], 
                new_col = 'propotion', 
                thinking = True)

    print(len(merged_df))
    print(merged_df)

    merged_df['propotion'] = pd.to_numeric(merged_df['propotion'], errors='coerce')
    prediction = merged_df['propotion'].mean().round(3)
    print(prediction)
    return prediction, op1.total_tokens + op2.total_tokens


# disney
# impute + match
def pipeline_48(args):
    query = "List the characters released in the year with highest ratio of Studio Entertainment revenue to total revenue."
    answer = ["Pocahontas"]
    characters = pd.read_csv("./databases/disney/characters.csv")
    revenue = pd.read_csv("./databases/disney/revenue.csv")
    revenue = revenue.dropna(subset=["Studio Entertainment[NI 1]", "Total"])
    op1 = LogicalImpute(OperandType.COLUMN)
    revenue = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the ratio of Studio Entertainment revenue to total revenue",
        df=revenue,
        depend_on=["Studio Entertainment[NI 1]", "Total"],
        new_col="ratio",
        thinking=True,
    ) 
    revenue = revenue.sort_values("ratio", ascending=False)[:1]

    print(revenue)  # 1995
    op2 =LogicalMatch(OperandType.CELL)
    merged = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The release date corresponds to the year",
        left_df=characters,
        right_df=revenue,
        left_on="release_date",
        right_on="Year",
        thinking=True,
    )
    print(merged)
    return merged[["left_hero"]].values.tolist(), op1.total_tokens + op2.total_tokens



# nextiaJD
# select + match
def pipeline_49(args):
    query = "How many Theatre/Performance cultural spaces with over 200 seats are located within a 20-minute walking distance of a nearby community center, where both the cultural spaces and community centers are situated at a latitude higher than Vancouver City Hall?"
    answer = 17
    cultural_spaces = pd.read_csv("./databases/nextiaJD/cultural-spaces.csv")
    cultural_spaces = cultural_spaces[cultural_spaces["TYPE"] == "Theatre/Performance"]
    cultural_spaces = cultural_spaces[cultural_spaces["NUMBER_OF_SEATS"] > 200]
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    cultural_spaces = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This location has a higher latitude than Vancouver City Hall",
        df=cultural_spaces,
        depend_on="Geom",
        thinking=True,
    )
    print(cultural_spaces["ADDRESS"])  # 63

    community_centres = pd.read_csv("./databases/nextiaJD/community-centres.csv")
    community_centres = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This location has a higher latitude than Vancouver City Hall",
        df=community_centres,
        depend_on="Geom",
        thinking=True,
    )
    print(community_centres["ADDRESS"])  # 6

    op2 = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The two addresses are within a 20-minute walk",
        left_df=cultural_spaces,
        right_df=community_centres,
        left_on="ADDRESS",
        right_on="ADDRESS",
        thinking=True,
    )
    merged_df = merged_df[['left_ADDRESS']].drop_duplicates()
    # print(merged_df[['left_ADDRESS', 'right_ADDRESS']])
    return merged_df.shape[0], op1.total_tokens + op2.total_tokens


# nextiaJD
# select + select + order
def pipeline_50(args):
    query = "List the Top 1 most-played Taylor Swift songs about love, from Taylor Swift's full-length studio albums in the iTunes table."
    answer = ['Blank Space']

    music = pd.read_csv("./databases/music/itunes.csv")
    music = music[(music["Artist_Name"] == "Taylor Swift")]
    print(music.shape)  # 194
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    music = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The album is an official full-length studio album by Taylor Swift.",
        df=music,
        depend_on="Album_Name",
        thinking=True,
    )
    print(music.shape)

    music = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The song is mainly about love.",
        df=music,
        depend_on="Song_Name",
        thinking=True,
    )
    print(music.shape)

    op2 = LogicalOrder(operand_type=OperandType.ROW)
    music = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The song's total play count",
        df=music,
        depend_on="Song_Name",
        k=1,
        ascending=False,
        thinking=True,
    )
    return music["Song_Name"].tolist(), op1.total_tokens + op2.total_tokens


# movies
# select + select
def pipeline_51(args):
    query = "Among movies with rating more than 7.5 in the 'rotten_tomatoes' table, how many of them are released in the same year as Christopher Nolan's Inception, were directed by female directors?"
    answe = 2
    rotten_tomatoes = pd.read_csv("./databases/movies/rotten_tomatoes.csv")
    rotten_tomatoes["Rating"] = pd.to_numeric(
        rotten_tomatoes["Rating"], errors="coerce"
    )
    rotten_tomatoes = rotten_tomatoes[rotten_tomatoes["Rating"] > 7.5]
    print(rotten_tomatoes.shape)  # 220
    op = LogicalSelect(operand_type=OperandType.ROW)
    rotten_tomatoes = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The movie is released in the same year as Christopher Nolan's Inception",
        df=rotten_tomatoes,
        depend_on="Year",
        thinking=True,
    )

    rotten_tomatoes = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The director is a female director",
        df=rotten_tomatoes,
        depend_on="Director",
        thinking=True,
    )
    return rotten_tomatoes.shape[0], op.total_tokens



# movie_3
# select + cluster
def pipeline_52(args):
    query = "Consider English Action movies rated NC-17 with a 'Reflection' theme. Group them into 'Comedy', 'Thriller', 'Tragedy' and 'Others'. Count the movies within each group."
    answer = [['Comedy', 2], ['Thriller', 4]]
    film = pd.read_csv("./databases/movie_3/film.csv")
    film_category = pd.read_csv("./databases/movie_3/film_category.csv")
    category = pd.read_csv("./databases/movie_3/category.csv")
    language  = pd.read_csv("./databases/movie_3/language.csv")
    language = language.rename(columns={"name": "language_name", "last_update": "language_last_update"})

    language = language[language['language_name'] == 'English']

    film = film[film["rating"] == "NC-17"]
    print(film.shape[0])  # 194
    
    merged = pd.merge(film, film_category, on="film_id")
    merged = pd.merge(merged, category, on="category_id")
    merged = merged[merged["name"] == "Action"]
    print(merged["description"])  # 12
    print(merged.columns)

    op1 = LogicalSelect(operand_type=OperandType.ROW)
    reflection_films = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The movie is described as a Reflection",
        df=merged,
        depend_on=["description"],
        thinking=True,
    )
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 0)

    print(reflection_films[['title', 'description']])
    reflection_films = pd.merge(reflection_films, language, on="language_id")

    op2 = LogicalCluster(operand_type=OperandType.ROW)
    result = op2.execute(impl_type=ImplType.LLM_ALL,
                                condition="Group the films into 'Comedy', 'Thriller', 'Tragedy' and 'Others'",
                                df=reflection_films,
                                depend_on=['title', 'description'],
                                thinking=True)

    result = result.groupby("cluster_name").agg(categoryCount=("film_id", "count")).reset_index()
    
    print(result)
    prediction = result.values.tolist()
    return prediction, op1.total_tokens + op2.total_tokens
 


# mondial_geo
# select + select
def pipeline_53(args):
    query = "List the top 3 European Union countries with the highest population growth that have a republican government."
    answer = ['Reunion', 'Luxembourg', 'Cyprus']
    politics = pd.read_csv("./databases/mondial_geo/politics.csv")
    population = pd.read_csv("./databases/mondial_geo/population.csv")
    country = pd.read_csv("./databases/mondial_geo/country.csv")

    merged = pd.merge(politics, population, on="Country")
    merged = pd.merge(merged, country, left_on="Country", right_on="Code")
    
    print(merged['Name'])
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    merged = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The country is a European Union country",
        df=merged,
        depend_on=["Name"],
        thinking=True,
    )
    
    print(merged)
    
    politics = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The goverment is a republican government",
        df=merged,
        depend_on=["Government"],
        thinking=True,
    )    
    
    print(merged['Name'])
    print(merged.shape[0])  # 13
    merged = merged.sort_values(by="Population_Growth", ascending=False)[:3]
    return merged["Name"].tolist(), op.total_tokens


# mondial_geo
# select + impute
def pipeline_54(args):
    query = "Among European countries with more than 200,000 GDP, which one has the lowest Population Density?"
    answer = "Russia"
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    economy = pd.read_csv("./databases/mondial_geo/economy.csv")
    merged = pd.merge(country, economy, left_on="Code", right_on="Country")
    merged = merged[merged["GDP"] > 200000]
    print(merged.shape[0])

    print(merged)
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    merged = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The country is a European country",
        df=merged,
        depend_on=["Name"],
        thinking=True,
    )
    print(merged)
    print(merged.shape[0])
    op2 = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The population density. Answer ONLY with a float number.",
        df=merged,
        depend_on=["Area", "Population"],
        new_col="population_density",
        thinking=True,
    )
    result['population_density'] = pd.to_numeric(result['population_density'])
    print(result)
    result = result.sort_values(by="population_density", ascending=True)[:1]
    return result["Name"].tolist(), op1.total_tokens + op2.total_tokens


# hockey
# impute + impute
def pipeline_55(args):
    query = "Calculate the average age (as of 2025) of coaches who are still alive, whose bmi (based on the 'height' in inches and the 'weight' in pounds in the 'Matser' Table) is higher than 28 who have been a player."
    answer = 74.29
    master = pd.read_csv("./databases/hockey/Master.csv")
    master = master.dropna(subset=["playerID", "coachID"])
    # print(master.shape[0]) # 268
    # master['bmi'] = master['weight'] * 1.0 / master['height'] / master['height'] * 703
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    master = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the BMI of each player based on the 'height' in inches and the 'weight' in pounds. Answer with ONLY the BMI value.",
        df=master,
        depend_on=["height", "weight"],
        new_col="bmi",
        thinking=True,
    )
    master["bmi"] = pd.to_numeric(master["bmi"])
    master = master[master["bmi"] > 28]
    print(master.shape[0])  # 12
    master = master[master["deathYear"].isna()]
    print(master[['birthYear', 'deathYear']])
    
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    master = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the age of each player as of 2025. Answer with ONLY the age value.",
        df=master,
        depend_on=["birthYear"],
        new_col="age",
        thinking=True,
    )
    master["age"] = pd.to_numeric(master["age"])
    
    return master["age"].mean().round(2), op.total_tokens


# hockey
# select + select
def pipeline_56(args):
    query = "Among the coaches who have served a team named with a city, how many of them are born in Northeastern States in USA?"
    answer = 16
    coaches = pd.read_csv("./databases/hockey/Coaches.csv")
    teams = pd.read_csv("./databases/hockey/Teams.csv")
    master = pd.read_csv("./databases/hockey/Master.csv")

    coaches = coaches[["coachID", "tmID"]].drop_duplicates()
    master = master[master["birthCountry"] == "USA"]

    unique_team_id = coaches["tmID"].drop_duplicates()
    teams = teams[["tmID", "name"]].drop_duplicates()
    teams = pd.merge(unique_team_id, teams, on="tmID")
    print(teams.shape[0])  # 114
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    teams = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The team name includes a city name",
        df=teams,
        depend_on=["name"],
        thinking=True,
    )
    # print(teams.shape[0]) # 104
    merged = pd.merge(coaches, teams, on="tmID")
    merged = pd.merge(merged, master, on="coachID")
    # print(merged.shape[0]) # 52
    print(merged["birthState"])
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The birth state in Northeastern States of USA",
        df=merged,
        depend_on=["birthState"],
        thinking=True,
    )
    result = result[['coachID']].drop_duplicates()
    return result.shape[0], op.total_tokens


# formula_1
# select + match
def pipeline_57(args):
    query = "Among the drivers who raced in Europe in 2015, how many of them have at least a circuit in their hometown country?"
    answer = 17
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")
    races = races[races["year"] == 2015]
    # print(races.shape[0]) # 19
    op1 = LogicalSelect(operand_type=OperandType.ROW)
    races = op1.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The race happens in Europe",
        df=races,
        depend_on=["name"],
        thinking=True,
    )
    # print(races.shape[0]) # 8
    merged = pd.merge(races, results, on="raceId")
    merged = pd.merge(merged, circuits, on="circuitId")
    merged = pd.merge(merged, drivers, on="driverId")
    # print(merged.shape[0]) # 160
    merged = merged[
        ["driverRef", "forename", "surname", "nationality"]
    ].drop_duplicates()
    print(merged)  # 20

    circuits = circuits[["country"]].drop_duplicates()
    print(circuits)

    op2 = LogicalMatch(operand_type=OperandType.CELL)
    drivers = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The nationality matches the country",
        left_df=merged,
        right_df=circuits,
        left_on="nationality",
        right_on="country",
        thinking=True,
    )
    drivers = drivers[['left_driverRef']].drop_duplicates()
    return drivers.shape[0], op1.total_tokens + op2.total_tokens


# formula_1
# match + order
def pipeline_58(args):
    query = "Identify the countries of the top 2 drivers with the highest total points. Rank the racing circuits in these countries by their first year to host a Formula 1 Grand Prix, from the earliest to the latest."
    answer = ['Silverstone Circuit', 'Nürburgring', 'Aintree', 'AVUS', 'Brands Hatch', 'Hockenheimring', 'Donington Park']
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")

    grouped = results.groupby("driverId").agg({"points": "sum"}).reset_index()
    grouped = grouped.sort_values("points", ascending=False)[:2]
    merged = pd.merge(drivers, grouped, on="driverId")

    op1 = LogicalMatch(operand_type=OperandType.CELL)
    circuits = op1.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The nationality matches the country",
        left_df=merged,
        right_df=circuits,
        left_on="nationality",
        right_on="country",
        thinking=True,
    )
    print(circuits)
    
    op2 = LogicalOrder(operand_type=OperandType.ROW)
    circuits = op2.execute(
        impl_type=ImplType.LLM_ALL,
        condition="the circuit's first year to host a Formula 1 Grand Prix",
        df=circuits,
        k=len(circuits),
        ascending=True,
        depend_on=["right_name", "right_location"],
        thinking=True,
    )
    return circuits["right_name"].tolist(), op1.total_tokens + op2.total_tokens


# formula_1
# match + impute 
def pipeline_59(args):
    query = "Identify the countries represented by constructors in the 2015 Formula 1 World Championship. Among these countries, list the country that hosted the highest-altitude Grand Prix that year."
    answer = ['Austria']
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")

    races = races[races["year"] == 2015]
    merged = pd.merge(races, results, on="raceId")
    merged = pd.merge(merged, constructors, on="constructorId")
    # print(merged.shape[0]) # 378
    merged = merged[["nationality"]].drop_duplicates()
    # print(merged.shape[0]) # 6

    circuits = circuits.rename(columns={"name": "circuit_name"})
    races = pd.merge(races, circuits, on="circuitId")
    
    op1 = LogicalMatch(operand_type=OperandType.CELL)
    races = op1.execute(
        impl_type=ImplType.LLM_ALL,
        condition="The nationality matches the race name",
        left_df=merged,
        right_df=races,
        left_on="nationality",
        right_on="name",
        thinking = True,
    )
    print(races)

    op2 = LogicalImpute(operand_type=OperandType.COLUMN)
    races = op2.execute(
        impl_type=ImplType.LLM_ONE,
        condition="the altitude of the race. Answer with ONLY the altitude value.",
        df=races,
        depend_on="right_circuit_name",
        new_col="altitude",
        thinking=True,
    )
    races['altitude'] = pd.to_numeric(races['altitude'])
    races = races.sort_values("altitude", ascending= False)[:1]
    return races["right_country"].tolist(), op1.total_tokens + op2.total_tokens


# beer_factory
# select + select
def pipeline_60(args):
    query = "Among male customers in Sacramento using Microsoft email services, count how many distinct beer brands they have purchased, limiting to brands first brewed after World War II."
    answer = 13
    customer = pd.read_csv("./databases/beer_factory/customers.csv")
    transaction = pd.read_csv("./databases/beer_factory/transaction.csv")
    beer_brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    beer = pd.read_csv("./databases/beer_factory/rootbeer.csv")

    customer = customer[
        (customer["City"] == "Sacramento") & (customer["Gender"] == "M")
    ]
    op = LogicalSelect(operand_type=OperandType.ROW)
    filtered_customer = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This email address uses Microsoft Service.",
        df=customer,
        depend_on=["Email"],
        thinking=True,
    )

    filtered_beer_brand = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This year is later than the end of World War II.",
        df=beer_brand,
        depend_on=["FirstBrewedYear"],
        thinking=True,
    )

    merged_df = pd.merge(filtered_customer, transaction, on="CustomerID")
    merged_df = pd.merge(merged_df, beer, on="RootBeerID")
    merged_df = pd.merge(merged_df, filtered_beer_brand, on="BrandID")

    merged_df = merged_df[["BrandName"]].drop_duplicates()

    prediction = len(merged_df)
    return prediction, op.total_tokens


