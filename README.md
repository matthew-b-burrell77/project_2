# Project 2 - Ames Housing Data and Kaggle Challenge

# Problem Statement

I signed up for an Ames housing data challenge on Kaggle. The challenge is to create a regression model to predict a price of a home. The Ames Housing Dataset is an exceptionally detailed and robust dataset with over 70 columns of different features relating to houses. The model used to predict prices of homes will be an OLS (Ordinary Least Squared) Model. OLS is a white box model, which means I can use coefficents form the model to gauge what features effect the price of a home. The success of the model will be determined by overall preformance of prediction on data the model was not trained on. If successful, the model could be used by sellers and buyers to determine a fair market price for homes on the market. Also, the model could be used by home flippers to focus on upgrading certain features to increase the price of a home to make a better profit when they sell. 


# Data description

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**PID**|int64| Train/Test |Parcel identification number  - can be used with city web site for parcel review.|
|**MS SubClass**| int64 |Train/Test|Identifies the type of dwelling involved in the sale.	|
|**MS Zoning**| object| Train/Test | Identifies the general zoning classification of the sale. |
|**Lot Frontage** | float64 | Train/Test | Linear feet of street connected to property |
|**Lot Area** |int64|Train/Test|Lot size in square feet|
|**Street**| object | Train/Test | Type of road access to property |
|**Alley**| object | Train/Test | Type of alley access to property |
|**Lot Shape** | object | Train/Test | General shape of property |
|**Land Contour** | object | Train/Test |Flatness of the property| 
|**Utilities**| object | Train/Test | Type of utilities available|
|**Lot Config** | object | Train/Test | Lot configuration|
|**Land Slope**| object | Train/Test |Slope of property |
|**Neighborhood**| object | Train/Test | Physical locations within Ames city limits|
|**Condition 1**| object | Train/Test | Proximity to various conditions|
|**Condition 2**| object | Train/Test | Proximity to various conditions (if more than one is present)|
|**Bldg Type** | object | Train/Test | Type of dwelling |
| **House Style** | object | Train/Test | Style of dwelling |
| **Overall Qual** | int64 | Train/Test | Rates the overall material and finish of the house |
|**Overall Cond** | int64 | Train/Test | Rates the overall condition of the house |
|**Year Built** | int64 | Train/Test | Original construction date |
|**Year Remod/Add** | int64 | Train/Test | Remodel date (same as construction date if no remodeling or additions) |
|**Roof Style**| object | Trian/Test | Type of roof |
|**Roof Matl** | object | Train/Test | Roof material |
|**Exterior 1st**| object | Train/Test |Exterior covering on house|
|**Exterior 2nd**| object | Train/Test | Exterior covering on house (if more than one material) | 
|**Mas Vnr Type** | object | Train/Test | Masonry veneer type|
|**Mas Vnr Area** | float64 | Train/Test | Masonry veneer area in square feet|
|**Exter Qual** | object | Train/Test | Evaluates the quality of the material on the exterior|
|**Exter Cond**| object | Train/Test | Evaluates the present condition of the material on the exterior |
|**Foundation**| object | Train/Test | Type of foundation |
|**Bsmt Qual**| object | Train/Test | Evaluates the height of the basement|
|**Bsmt Cond**|object|Train/Test|Evaluates the general condition of the basement|
|**Bsmt Exposure**|object|Train/Test| Refers to walkout or garden level walls|
|**BsmtFin Type 1**|object| Train/Test|Rating of basement finished area|
|**BsmtFin SF 1**|float64|Train/Test|Type 1 finished square feet|
|**BsmtFinType 2**|object|Train/Test|Rating of basement finished area (if multiple types)|
|**BsmtFin SF 2**|float64|Train/Test|Type 2 finished square feet|
|**Bsmt Unf SF**|float64|Train/Test|Unfinished square feet of basement area|
|**Total Bsmt SF**|float64|Train/Test|Total square feet of basement area|
|**Heating**|object|Train/Test| Type of heating|
|**HeatingQC**|object|Train/Test| Heating quality and condition|
|**Central Air**|object|Train/Test| Central air conditioning|
|**Electrical**|object|Train/Test| Electrical system |
|**1st Flr SF**|int64|Train/Test| First Floor square feet|
|**2nd Flr SF**|int64|Train/Test| Second floor square feet|
|**Low Qual Fin SF**|int64|Train/Test| Low quality finished square feet (all floors)|
|**Gr Liv Area**|int64|Train/Test| Above grade (ground) living area square feet|
|**Bsmt Full Bath**|float64|Train/Test| Basement full bathrooms|
|**Bsmt Half Bath**|float64|Train/Test| Basement half bathrooms|
|**Full Bath**|int64|Train/Test|Full bathrooms above grade|
|**Half Bath**|int64|Train/Test| Half baths above grade|
|**Bedroom**|int64|Train/Test| Bedrooms above grade (does NOT include basement bedrooms)|
|**Kitchen**|int64|Train/Test| Kitchens above grade|
|**KitchenQual**|object|Train/Test|Kitchen quality|
|**TotRmsAbvGrd**|int64|Train/Test| Total rooms above grade (does not include bathrooms)|
|**Functional**|object|Train/Test|Home functionality (Assume typical unless deductions are warranted)|
|**Fireplaces**|int64|Train/Test|Number of fireplaces|
|**FireplaceQu**|object|Train/Test|Fireplace quality|
|**Garage Type**|object|Train/Test|Garage location|
|**Garage Yr Blt**|float64|Train/Test| Year garage was built|
|**Garage Finish**|object|Train/Test| Interior finish of the garage|
|**Garage Cars**|float64|Train/Test|Size of garage in car capacity|
|**Garage Area**|float64|Train/Test|Size of garage in square feet|
|**Garage Qual**|object|Train/Test|Garage quality|
|**Garage Cond**|object|Train/Test|Garage condition|
|**Paved Drive**|object|Train/Test|Paved driveway|
|**Wood Deck SF**|int64| Train/Test|Wood deck area in square feet|
|**Open Porch SF**|int64|Train/Test|Open porch area in square feet|
|**Enclosed Porch**|int64|Train/Test|Enclosed porch area in square feet|
|**3-Ssn Porch**|int64|Train/Test|Three season porch area in square feet|
|**Screen Porch**|int64|Train/Test|Screen porch area in square feet|
|**Pool Area**|int64|Train/Test|Pool area in square feet|
|**Pool QC**|object|Train/Test|Pool quality|
|**Fence**|object|Train/Test|Fence quality|
|**Misc Feature**|object|Train/Test|Miscellaneous feature not covered in other categories|
|**Misc Val**|int64|Train/Test|$Value of miscellaneous feature|
|**Mo Sold**|int64|Train/Test|Month Sold (MM)|
|**Yr Sold**|int64|Train/Test|Year Sold (YYYY)|
|**Sale Type**|object|Train/Test|Type of sale|
|**Sale Condition**|object|Train/Test|Condition of sale|
|**SalePrice**|int64|Train|Sale price $$|

MS SubClass:

       020	1-STORY 1946 & NEWER ALL STYLES
       030	1-STORY 1945 & OLDER
       040	1-STORY W/FINISHED ATTIC ALL AGES
       045	1-1/2 STORY - UNFINISHED ALL AGES
       050	1-1/2 STORY FINISHED ALL AGES
       060	2-STORY 1946 & NEWER
       070	2-STORY 1945 & OLDER
       075	2-1/2 STORY ALL AGES
       080	SPLIT OR MULTI-LEVEL
       085	SPLIT FOYER
       090	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES
       
MS Zoning:

       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density

Street:

       Grvl	Gravel	
       Pave	Paved
       	
Alley:

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
       
Lot Shape:

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
       
Land Contour:

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression 

Utilities: 
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
       
Land Slope:
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
       
Neighborhood:

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       Greens	Greens
       GrnHill	Green Hills
       IDOTRR	Iowa DOT and Rail Road
       Landmrk	Landmark
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
       
Condition 1:
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Condition 2: 
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Bldg Type:
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
House Style:
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
	
Overall Qual: 

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
Overall Cond:

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
       
Roof Style:

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
		
Roof Matl:

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
Exterior 1st:

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
Exterior 2nd: 

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
       
Exter Qual:
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Exter Cond:
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Foundation: 

       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood

Bsmt Qual:

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
		
Bsmt Cond:

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
	
Bsmt Exposure:

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
	
BsmtFin Type 1:

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinType 2:

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
       

Heating:
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
		
HeatingQC:

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Central Air:

       N	No
       Y	Yes
		
Electrical:

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       
KitchenQual:

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       
Functional: 

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
       
Fireplace Qu:

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
		
Garage Type:
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
       
Garage Finish:

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
	
Garage Qual: 

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
Garage Cond: 

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
Paved Drive: 

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
 
Pool QC:
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
		
Fence:
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
	
Misc Feature:
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
       
Sale Type:
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
		
Sale Condition:

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)
       
Mas Vnr Type:

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	    None
       Stone	Stone

**Conclusion and Recommendations**
1. The OLS model does a fair job with predicting sale prices of homes
    * train score 0.8442
    * test score 0.8487 
    * cross val score 0.8222
2. To increase the price of a home the best bet is to improve condition of the home 
    * for exmaple: homes that are in excellent condition, there are 10.2% more valuable 
3. Next steps to improve the model 
    * adding more features from ames data
    * Scale features 
    * Using other machine learning models like lasso or ridge regression
