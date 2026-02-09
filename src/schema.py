from pydantic import BaseModel, Field
from typing import List

class CreditRequest(BaseModel):
    """
    Credit application request with raw UCI dataset fields.
    All bill and payment fields are for the last 6 months.
    """
    # Core demographics
    LIMIT_BAL: float = Field(..., description="Credit limit (income proxy)")
    AGE: int = Field(..., gt=18, lt=100, description="Age in years")
    SEX: int = Field(..., ge=1, le=2, description="Gender (1=male, 2=female)")
    EDUCATION: int = Field(..., ge=0, le=6, description="Education level (0-6)")
    MARRIAGE: int = Field(..., ge=0, le=3, description="Marital status (0-3)")
    
    # Payment status for last 6 months (-2 to 8, where -1=pay duly, 1=delay 1 month, etc.)
    PAY_0: int = Field(..., description="Repayment status in September")
    PAY_2: int = Field(..., description="Repayment status in August")
    PAY_3: int = Field(..., description="Repayment status in July")
    PAY_4: int = Field(..., description="Repayment status in June")
    PAY_5: int = Field(..., description="Repayment status in May")
    PAY_6: int = Field(..., description="Repayment status in April")
    
    # Bill amounts for last 6 months
    BILL_AMT1: float = Field(..., description="Bill statement in September")
    BILL_AMT2: float = Field(..., description="Bill statement in August")
    BILL_AMT3: float = Field(..., description="Bill statement in July")
    BILL_AMT4: float = Field(..., description="Bill statement in June")
    BILL_AMT5: float = Field(..., description="Bill statement in May")
    BILL_AMT6: float = Field(..., description="Bill statement in April")
    
    # Payment amounts for last 6 months
    PAY_AMT1: float = Field(..., ge=0, description="Payment in September")
    PAY_AMT2: float = Field(..., ge=0, description="Payment in August")
    PAY_AMT3: float = Field(..., ge=0, description="Payment in July")
    PAY_AMT4: float = Field(..., ge=0, description="Payment in June")
    PAY_AMT5: float = Field(..., ge=0, description="Payment in May")
    PAY_AMT6: float = Field(..., ge=0, description="Payment in April")
    
    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 200000,
                "AGE": 35,
                "SEX": 1,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": -1,
                "PAY_6": -1,
                "BILL_AMT1": 50000,
                "BILL_AMT2": 48000,
                "BILL_AMT3": 45000,
                "BILL_AMT4": 47000,
                "BILL_AMT5": 46000,
                "BILL_AMT6": 44000,
                "PAY_AMT1": 2000,
                "PAY_AMT2": 2500,
                "PAY_AMT3": 2200,
                "PAY_AMT4": 2300,
                "PAY_AMT5": 2100,
                "PAY_AMT6": 2000
            }
        }
