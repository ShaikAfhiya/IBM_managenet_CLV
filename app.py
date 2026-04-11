from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel,Field
import joblib
#loading model
model=joblib.load('ibm_clv.pkl')
#fastapi
app=FastAPI()
class Data(BaseModel):
     responce:str=Field(alias='Response')
     coverage:str=Field(alias='Coverage')
     education:str=Field(alias='Education')
     employmentStatus:str=Field(alias='EmploymentStatus')
     gender:str=Field(alias='Gender')
     income:int=Field(alias='Income')
     Location_Code:str=Field(alias="Location Code")
     Marital_Status:str=Field(alias="Marital Status")
     Monthly_Premium_Auto:int=Field(alias='Monthly Premium Auto')
     Months_Since_Last_Claim:int=Field(alias='Months Since Last Claim')
     Months_Since_Policy_Inception:int=Field(alias='Months Since Policy Inception')
     Number_of_Open_Complaints:int=Field(alias='Number of Open Complaints')
     Number_of_Policies:int=Field(alias='Number of Policies')
     Policy_Type:str=Field(alias='Policy Type')
     Policy:str=Field(alias='Policy')
     Renew_Offer_Type:str=Field(alias='Renew Offer Type')
     Sales_Channel:str=Field(alias='Sales Channel')
     Total_Claim_Amount:float=Field(alias='Total Claim Amount')
     Vehicle_Class:str=Field(alias='Vehicle Class')
     Vehicle_Size:str=Field(alias='Vehicle Size')
     class Config:
        populate_by_name = True  # IMPORTANT
@app.post('/')
def predict(data:Data):
      input_dict = data.dict(by_alias=True)
      df=pd.DataFrame([input_dict])
      prediction=float(model.predict(df)[0])
      return{
        'Customer Lifetime Value Prediction':prediction
    }
