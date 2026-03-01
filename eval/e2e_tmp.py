import pandas as pd
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import (
    LogicalMap,
    LogicalImpute,
    LogicalGroupBy,
    LogicalOrder,
)
from src.core.enums import OperandType, ImplType

# Match + Match 3/3
# Match + Impute 3/3
# Match + Cluster 3/3
# Match + Order 3/3

# Cluster + Cluster 0/3
# Cluster + Order 3/3: 筛选五名最高的人后，根据国籍进行聚类
# Order + Order 0/3: 根据不同指标分别筛选

# Match + Match
def pipeline_1000():
    query = ""
    answer = [3, 99]
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")
    
    drivers = drivers.dropna(subset=["dob"])
    drivers = drivers[drivers["dob"].str[:4].astype(int) >= 1992]
    circuits = circuits[circuits["lat"] >= 50]
    
    print(drivers[['forename', 'surname', 'nationality']])
    print(circuits[['name', 'country', 'lat']])
    print(constructors)
    
    op = LogicalMap(OperandType.CELL)
    drivers_circuits = op.execute(
        impl_type=ImplType.LLM_ONLY,
        condition="The nationality matches the country",
        left_df=drivers,
        right_df=circuits,
        left_on="nationality",
        right_on="country",
        thinking=True,
    )
    
    op = LogicalMap(OperandType.CELL)
    constructors_circuits = op.execute(
        impl_type=ImplType.LLM_ONLY,
        condition="The nationality matches the country",
        left_df=constructors,
        right_df=circuits,
        left_on="nationality",
        right_on="country",
        thinking=True,
    )
    
    drivers_circuits = drivers_circuits[['left_driverRef']].drop_duplicates()
    constructors_circuits = constructors_circuits[['left_constructorRef']].drop_duplicates()
    prediction = [len(drivers_circuits), len(constructors_circuits)]
    
    return prediction

def pipeline_1001():
    # How many islands are located nearby a ocean adjoins any countries with 50Million population.
    query = ""
    answer = ""
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    island = pd.read_csv("./databases/mondial_geo/island.csv")
    
    sea = sea.rename(columns={'Name': 'sea_name'})
    country = country.rename(columns={'Name': 'country_name'})
    country = country[country['Population'] > 50000000]
    
    print(sea)
    print(country)
    
    op = LogicalMap(OperandType.CELL)
    sea_country = op.execute(impl_type=ImplType.LLM_ONLY,
                            condition="The sea adjoins the country.",
                            left_df=sea,
                            right_df=country,
                            left_on='sea_name',
                            right_on='country_name',
                            thinking = True) 
    
    merged_df = op.execute(impl_type=ImplType.LLM_ONLY,
                            condition="The island adjoins the sea.",
                            left_df=island,
                            right_df=sea_country,
                            left_on='Islands',
                            right_on='left_sea_name',
                            thinking = True) 
    
    print(merged_df)
    
    prediction = len(merged_df[['left_Islands']].drop_duplicates())
    return prediction
    

def pipeline_1002():
    query = ""
    answer = ['1404', '1418', '32703', '59885']
    comments = pd.read_csv("./databases/codebase_community/comments.csv")
    tags = pd.read_csv("./databases/codebase_community/tags.csv")
    badges = pd.read_csv("./databases/codebase_community/badges.csv")
        
    comments = comments[comments['Score'] > 20]
    comments = comments[['Id', 'Text', 'Score']]
    tags = tags.sort_values(by='Count', ascending=False)[5:15]
    top_badges = badges.drop_duplicates(subset=["Name"]).sort_values("Name", ascending=False).head(10)
    
    print(top_badges)

    print(comments)
    print(tags)
    
    op = LogicalMap(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_SEMI,
                            condition="The content of the comment 'Text' strictly contains the entire or part of the 'TagName'.", 
                            left_df=comments, 
                            right_df=tags, 
                            left_on='Text', 
                            right_on='TagName',
                            thinking = True) 
    
    merged_df = op.execute(impl_type=ImplType.LLM_SEMI,
                            condition="The content of the comment 'Text' strictly contains the entire or part of the 'BadgeName'.", 
                            left_df=merged_df, 
                            right_df=top_badges, 
                            left_on='left_Text', 
                            right_on='Name',
                            thinking = True) 

    prediction = merged_df[['left_left_Id']].values.tolist()
    return prediction


# Match + Impute
def pipeline_1003():
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

    op = LogicalMap(OperandType.ROW)
    merged_df = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="The two senior government official names shares the same first name.",
        left_df=home_office_officials,
        right_df=travel_2018_officials,
        left_on="Name of Official",
        right_on="Senior Officials Name",
        thinking=False,
    )
    print(merged_df.columns)
    merged_df = merged_df[["right_Senior Officials Name", "right_Destination"]]

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="Impute the continent where this destination locates.",
        df=merged_df,
        depend_on="right_Destination",
        new_col="continent",
    )

    prediction = result.values.tolist()
    return prediction


# Match + Impute
def pipeline_1004():
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
    races = LogicalMap(operand_type=OperandType.CELL).execute(
        impl_type=ImplType.LLM_SEMI_OPTIM,
        condition="The nationality matches the race name",
        left_df=merged,
        right_df=races,
        left_on="nationality",
        right_on="name",
        thinking = True,
    )
    print(races)

    races = LogicalImpute(operand_type=OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_SEMI,
        condition="the altitude of the race",
        df=races,
        depend_on="right_circuit_name",
        new_col="altitude",
        thinking=True,
    )
    races = races.sort_values("altitude", ascending= False)[:1]
    return races["right_country"].tolist()


# Match + Impute
def pipeline_1005():
    query = "For the comments with a score above 30, identify those that discuss any of the top 10 most frequent tags. For each such comment, infer if the comment's text is related to statistics(answer only with 'True' or 'False')."
    answer = [[1915, 'True'], [2027, 'True'], [167835, 'True'], [175302, 'True']]
    comments = pd.read_csv("./databases/codebase_community/comments.csv")
    tags =  pd.read_csv("./databases/codebase_community/tags.csv")
    comments = comments[comments['Score'] > 30]
    
    comments = comments[['Id', 'Text', 'Score']]
    print(len(comments))
    tags = tags.sort_values(by='Count', ascending=False)[:10]

    op = LogicalMap(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_SEMI,
                            condition="The content of the comment 'Text' is related to the 'TagName'", 
                            left_df=comments, 
                            right_df=tags, 
                            left_on='Text', 
                            right_on='TagName',
                            thinking = True) 
    print(merged_df)
    merged_df = merged_df[['left_Id', 'left_Text']].drop_duplicates(subset=['left_Id'])
    
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    merged_df = op.execute(impl_type=ImplType.LLM_SEMI,
                            condition="Is this comment text related to statistics? Answer ONLY with 'True' or 'False'.",
                            df=merged_df,
                            depend_on='left_Text',
                            new_col='is_related_to_statistics',
                            thinking = True)
    
    prediction = merged_df[['left_Id','is_related_to_statistics']].values.tolist()
    return prediction


# Match + Cluster
def pipeline_1006():
    query = "List all unique countries present in the CIHR co-applicant table (2006-07) for which there are protected heritage areas found in the human-wildlife coexistence table. Answer with the unique country names."
    answer = [['Alberta', 5], ['British Columbia', 7], ['Manitoba', 4], ['New Brunswick', 2], ['Newfoundland and Labrador', 3], ['Northwest Territories', 2], ['Nova Scotia', 3], ['Nunavut', 1], ['Ontario', 7], ['Prince Edward Island', 1], ['Quebec', 5], ['Saskatchewan', 3], ['Yukon', 3]]

    cihr_data = pd.read_csv("./databases/santos/cihr_co-applicant_200607.csv")
    wildlife_data = pd.read_csv(
        "./databases/santos/pca-human-wildlife-coexistence-animals-involved-detailed-records.csv"
    )

    # Filter to get unique countries from both datasets
    cihr_countries = cihr_data[["CountryEN"]].drop_duplicates()
    wildlife_areas = wildlife_data[["Protected Heritage Area"]].drop_duplicates()

    print(len(cihr_countries), len(wildlife_areas))
    print(cihr_countries)
    print(wildlife_areas)

    op = LogicalMap(OperandType.ROW)
    merged_df = op.execute(
        impl_type=ImplType.LLM_SEMI_OPTIM,
        condition="The country name matches the protected heritage area location.",
        left_df=wildlife_areas,
        right_df=cihr_countries,
        left_on="Protected Heritage Area",
        right_on="CountryEN",
        thinking=True,
    )
    print(merged_df)
    
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="Cluster the data by the province or territories of Canada in which the national park locates.",
        df=merged_df,
        depend_on="left_Protected Heritage Area", 
        thinking = True,
    )
    print(result)
    prediction = result.groupby("cluster_name").agg({"left_Protected Heritage Area": "count"}).reset_index().values.tolist()
    
    return prediction


# Match + Cluster
def pipeline_1007():
    query = "Cluster Disney movie release dates from years when total revenue exceeded 35,000 into 'Leap Year' or 'Common Year', and count how many releases occurred in each category."
    answer = [['Common Year', 6], ['Leap Year', 4]]
    characters = pd.read_csv("./databases/disney/characters.csv")
    revenue = pd.read_csv("./databases/disney/revenue.csv")
    revenue = revenue[revenue['Total'] > 35000]
    print(len(characters), len(revenue))
    print(characters)
    print(revenue)

    op = LogicalMap(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_SEMI,
                            condition="The 'release_date' corresponds to the 'Year'", 
                            left_df=characters, 
                            right_df=revenue, 
                            left_on='release_date', 
                            right_on='Year',
                            thinking = True) 
    merged_df = merged_df[['left_release_date', 'right_Year']]
    
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="Cluster the data by the year to 'Leap Year' or 'Common Year'.",
        df=merged_df,
        depend_on="right_Year", 
        thinking = True,
    )
    prediction = result.groupby("cluster_name").agg({"right_Year": "count"}).reset_index().values.tolist()
    return prediction


# Match + Cluster
def pipeline_1008():
    query = "Filter constructors participated in 2016 Formula 1 world championship and list the number of constructors in each continent."
    answer = [['Asia', 1], ['Europe', 9], ['North America', 1]]
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")
    constructor_standings = pd.read_csv("./databases/formula_1/constructorStandings.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    
    merged = pd.merge(constructors, constructor_standings, on="constructorId")
    races = races[races["year"] == 2016] 
    merged = pd.merge(merged, races, on="raceId")
    
    merged = merged[["constructorId", "nationality"]].drop_duplicates()
    circuits = circuits[["country"]].drop_duplicates()
    
    print(merged)
    print(circuits)

    op = LogicalMap(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_ONLY,
                            condition="The nationality refers to the same country.",
                            left_df=merged,
                            right_df=circuits,
                            left_on="nationality",
                            right_on="country",
                            thinking = True)
    print(merged_df)
    
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    grouped_df = op.execute(
        impl_type=ImplType.LLM_ONLY,
        condition="Cluster the countries by continent.",
        df=merged_df,
        depend_on="right_country", 
        thinking = True,
    )
    prediction = grouped_df.groupby("cluster_name").agg({"left_constructorId": "count"}).reset_index().values.tolist()
    
    return prediction


# Match + Order
def pipeline_1009():
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

    op = LogicalMap(OperandType.ROW)
    merged_df = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="The two addresses represent the same street location.",
        left_df=pubs_addresses,
        right_df=community_addresses,
        left_on="address1",
        right_on="address1",
        thinking=True,
    )
    merged_df = merged_df.drop_duplicates()

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(merged_df)
    merged_df = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="the Latitude of street.",
        depend_on="right_address1",
        ascending=True,
        k=k,
        df=merged_df,
        sort_algo="heap",
    )

    prediction = merged_df.values.tolist()
    return prediction


# Match + Order
def pipeline_1010():
    query = "Identify the countries of the top 2 drivers with the highest total points. Rank the racing circuits in these countries by their construction year, from the earliest to the latest."
    answer = ['AVUS', 'Nürburgring', 'Silverstone Circuit', 'Brands Hatch', 'Aintree', 'Hockenheimring', 'Donington Park']
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")

    grouped = results.groupby("driverId").agg({"points": "sum"}).reset_index()
    grouped = grouped.sort_values("points", ascending=False)[:2]
    merged = pd.merge(drivers, grouped, on="driverId")

    circuits = LogicalMap(operand_type=OperandType.CELL).execute(
        impl_type=ImplType.LLM_SEMI,
        condition="The nationality matches the country",
        left_df=merged,
        right_df=circuits,
        left_on="nationality",
        right_on="country",
    )

    circuits = LogicalOrder(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_SEMI,
        condition="the construction year of the circuit",
        df=circuits,
        k=len(circuits),
        ascending=True,
        depend_on=["right_name", "right_location"],
    )
    return circuits["right_name"].tolist()


# Match + Order
def pipeline_1011():
    query = "For the top 5 customers by annual revenue, which ones are in a state that borders a state containing a city with over 600,000 people? Sort the result by latitude of the home state capital. List the customer ID, their home state, and the neighboring state."
    answer = [[954, 'LA'], [4426, 'LA'], [1724, 'OH'], [2421, 'WI']]
    customer = pd.read_csv("./databases/shipping/customer.csv")
    city = pd.read_csv("./databases/shipping/city.csv")
    
    customer = customer.sort_values(by='annual_revenue', ascending=False)[:5]
    customer = customer[['cust_id', 'cust_name', 'state']]
    city = city[city['population']>600000]
    city = city[['state']].drop_duplicates()
    
    op = LogicalMap(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_ONLY,
                            condition="The 'left_state' is adjacent to the 'right_state'", 
                            left_df=customer, 
                            right_df=city, 
                            left_on='state', 
                            right_on='state',
                            thinking = False) 
    merged_df = merged_df[['left_cust_id', 'left_state']].drop_duplicates()
    
    op = LogicalOrder(operand_type=OperandType.ROW)
    merged_df = op.execute(impl_type=ImplType.LLM_ONLY,
                            condition="the Latitude of captical city of the state.",
                            df=merged_df,
                            k=len(merged_df),
                            ascending=True,
                            depend_on=["left_state"],
                            sort_algo="heap",
                        )
    
    prediction = merged_df.values.tolist()
    return prediction



# Cluster + Cluster
def pipeline_1012():
    pass


# Cluster + Cluster
def pipeline_1013():
    pass


# Cluster + Cluster
def pipeline_1014():
    pass


# Cluster + Order
def pipeline_1015():
    query = "Among the 10 most expensive products with over 100 units in stock, count how many have a model year that is a leap year and how many have a model year that is a common year."
    answer = [['Common Year', 3], ['Leap Year', 2]]
    products = pd.read_csv("./databases/car_retails/products.csv")
    products = products[(products['quantityInStock'] > 80) & (products['buyPrice'] > 90)]
    
    print(products)
    
    op = LogicalOrder(operand_type=OperandType.ROW)
    products = op.execute(impl_type=ImplType.LLM_ONLY,
                        condition="The product year of this car.",
                        depend_on= 'productName',
                        ascending = True,
                        k = 5,
                        df=products, 
                        sort_algo='simple',
                        thinking = True)
    
    print(products)
    
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_SEMI,
                        condition="Cluster the year of the product into 'Leap Year' or 'Common Year'.", 
                        df = products,
                        depend_on = 'productName',
                        thinking = True)
    prediction = result.groupby("cluster_name").agg(productCount=("productCode", "count")).reset_index().values.tolist()              
    return prediction


# Cluster + Order
def pipeline_1016():
    query = "Classify circuits which host Grand Prix in 2016 by their location continent. Answer with the location and corresponding continent."
    answer = [['Europe', 7], ['North America', 3]]

    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    
    merged_df = pd.merge(
        circuits.rename(
            columns={"name": "circuit_name", "country": "location_country"}
        ),
        races,
        on="circuitId",
    )
    merged_df = merged_df[merged_df["year"] == 2016]
    print(merged_df['name'])
    
    op = LogicalOrder(operand_type=OperandType.ROW)
    merged_df = op.execute(impl_type=ImplType.LLM_ONLY,
                            condition="the first year of this grand prix",
                            df=merged_df,
                            k=10,
                            ascending=True,
                            depend_on=["name"],
                            sort_algo="heap",
                            thinking = True,
                        )
    
    print(merged_df)

    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="Cluster the data by the continent in which the circuit locates in",
        df=merged_df,
        depend_on="location_country", 
        thinking = True,
    )
    print(result.columns)
    result = result[["location", "cluster_name"]]
    prediction = result.groupby("cluster_name").agg({"location": "count"}).reset_index().values.tolist()

    return prediction


# Cluster + Order
def pipeline_1017():
    query = "Count how many Vintage Cars for sales are produced from 'USA', 'Europe' and 'Others' respectively."
    answer = [['Europe', 1], ['USA', 4]]
    products = pd.read_csv("./databases/car_retails/products.csv")
    productLines = pd.read_csv("./databases/car_retails/productlines.csv")

    productLines = productLines[productLines["productLine"] == "Vintage Cars"]
    merged_df = pd.merge(products, productLines, on="productLine")
    merged_df = merged_df[["productName"]]
    print(merged_df)
    
    op = LogicalOrder(operand_type=OperandType.ROW)
    merged_df = op.execute(impl_type=ImplType.LLM_ONLY,
                            condition="the founding year of the car construction company",
                            df=merged_df,
                            k=5,
                            ascending=True,
                            depend_on=["productName"],
                            sort_algo="heap",
                            thinking = True
                        )
    print(merged_df)

    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="Cluster the product data by their manufacture's country into 'USA', 'Europe', 'Others'.",
        df=merged_df,
        depend_on="productName",
    )

    result = (
        result.groupby("cluster_name").agg(count=("productName", "count")).reset_index()
    ).values.tolist()

    return result


# Order + Order
def pipeline_1018():
    query = "Rank Top 5 highest rated movies in imdb by their box office from highest to lowest, and return their ID accordingly."
    answer = ['a-569', 'a-1240', 'a-118', 'a-915', 'a-2007']
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    imdb = imdb.sort_values(by="Rating", ascending=False)[:10]
    print(imdb)
    print(imdb.columns)
    
    op = LogicalOrder(operand_type=OperandType.ROW)
    imdb = op.execute(impl_type=ImplType.LLM_ONLY,
                        condition="birtday of the leading actor",
                        depend_on="Cast",
                        ascending=True,
                        k=5,
                        df=imdb,
                        sort_algo="heap",
                        thinking = True
                    )

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(imdb)
    result = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="box office of this movie",
        depend_on="Title",
        ascending=False,
        k=k,
        df=imdb,
        sort_algo="heap",
    )
    result = result["ID"].tolist()

    return result


# Order + Order
def pipeline_1019():
    pass

# Order + Order
def pipeline_1020():
    pass


# movies
# cluster
def pipeline_86():
    query = "Among the top 10 highest-rated movies in imdb table, count how many of these films were directed by someone from each continent. What are the numbers for each continent?"
    answer = [['Europe Directors', 3], ['North America Directors', 6], ['Oceania Directors', 1]]
    movies = pd.read_csv("./databases/movies/imdb.csv")
    movies = movies.sort_values(by="Rating", ascending=False)[:10]
    movies = movies[['Title', 'Director']]
    print(movies.shape[0])

    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_SEMI,
                        condition="Group by the continent of the director's nationality", 
                        df = movies,
                        depend_on = 'Director',
                        thinking = True)
    print(result)
    prediction = result.groupby("cluster_name").agg(movieCount=("Title", "count")).reset_index().values.tolist()

    return prediction
   


# european_football_2
# order + order
def pipeline_88():
    query = "List the 5 tallest football players, and order them by the population size of their home countries, from largest to smallest"
    answer = ['Lacina Traore', 'Kristof van Hout', 'Nikola Zigic', 'Vanja Milinkovic-Savic', 'Bogdan Milic']
    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player = player.sort_values(by="height", ascending=False)[:5]
    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(player)
    result = op.execute(impl_type=ImplType.LLM_SEMI,
                        condition="the population of the home country of the football player listed in 'player_name' column.",
                        depend_on= 'player_name',
                        ascending = False,
                        k = k,
                        df=player, 
                        sort_algo='simple',
                        thinking = True)
    print(result)
    predication = result['player_name'].tolist()
    return predication
   

# mondial_geo
# order
def pipeline_89():
    query = "Among the Top 10 most populous countries, list those have an area greater than 1 million square kilometers, and rank them based on the location of their capital from the northest to the southest."
    answer = ['Russia', 'China', 'United States', 'India', 'Indonesia', 'Brazil']
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    country = country.sort_values(by="Population", ascending=False)[:10]
    print(country)
    country = country[country['Area'] > 1000000]
    print(country)

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(country)
    result = op.execute(impl_type=ImplType.LLM_SEMI,
                        condition="the latitude of capital of this country.",
                        depend_on= 'Name',
                        ascending = False,
                        k = k,
                        df=country, 
                        sort_algo='simple',
                        thinking = True)
    print(result)
    prediction = result['Name'].tolist()
    return prediction
    
# european_football_2
# order
def pipeline_93():
    query = "List leagues whose countries have Top 5 largest land area."
    answer = ['France Ligue 1', 'Spain LIGA BBVA', 'Germany 1. Bundesliga', 'Poland Ekstraklasa', 'Italy Serie A']
    country = pd.read_csv("./databases/european_football_2/Country.csv")
    leagues = pd.read_csv("./databases/european_football_2/League.csv")

    merged = pd.merge(
        leagues,
        country.rename(columns={"name": "country_name"}),
        left_on="country_id",
        right_on="id",
    )
    print(merged)
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="the land area of the country",
        depend_on="country_name",
        ascending=False,
        k=5,
        df=merged,
        sort_algo="simple",
        thinking=True,
    )
    return result["name"].tolist()


# mondial_geo
# cluster
def pipeline_110():
    query = "Among countries with population growth less than 1.0, group them by their continents, and then list the average GDP of 'Asia', 'Europe' and 'North America' accordingly."
    answer = ["856275", "213834", "411848"]
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    economy = pd.read_csv("./databases/mondial_geo/economy.csv")
    population = pd.read_csv("./databases/mondial_geo/population.csv")
    population = population[population["Population_Growth"] < 1.0]
    print(population.shape[0])
    merged = pd.merge(population, economy, on="Country")
    merged = pd.merge(merged, country, left_on="Country", right_on="Code")

    result = LogicalGroupBy(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_SEMI,
        condition="Cluster the countries by their continents into 'Asia', 'Europe', 'America' and 'Others'",
        df=merged,
        depend_on=["Name"],
        thinking = True,
    )
    result = result.groupby("cluster_name")["GDP"].mean()
    return result[["Asia", "Europe", "America"]].values



if __name__ == "__main__":
    start_time = time.time()
    print(pipeline_110())
    end_time = time.time()
    print(f"Time used for pipeline execution: {end_time - start_time} seconds")