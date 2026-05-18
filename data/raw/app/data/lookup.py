import os
import pandas as pd

class StructuredLookup:
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = raw_dir
        self.interest_rates_file = os.path.join(raw_dir, "InterestRate_FeeSchedule_NovaNorth.csv")
        self.loan_eligibility_file = os.path.join(raw_dir, "LoanEligibility_Matrix_NovaNorth.csv")
        
        self.interest_rates_df = None
        self.loan_eligibility_df = None
        self.load_data()
        
    def load_data(self):
        if os.path.exists(self.interest_rates_file):
            self.interest_rates_df = pd.read_csv(self.interest_rates_file)
        else:
            print(f"Warning: {self.interest_rates_file} not found.")
            
        if os.path.exists(self.loan_eligibility_file):
            self.loan_eligibility_df = pd.read_csv(self.loan_eligibility_file)
        else:
            print(f"Warning: {self.loan_eligibility_file} not found.")

    def lookup_interest_rate(self, query: str) -> str:
        if self.interest_rates_df is None:
            return "Interest rate data not available."
            
        # Identify product category from query
        query = query.lower()
        if "home" in query:
            matches = self.interest_rates_df[self.interest_rates_df['product_category'].str.contains("Home Loan", case=False)]
        elif "personal" in query:
            matches = self.interest_rates_df[self.interest_rates_df['product_category'].str.contains("Personal Loan", case=False)]
        elif "auto" in query or "car" in query:
            matches = self.interest_rates_df[self.interest_rates_df['product_category'].str.contains("Auto Loan", case=False)]
        elif "education" in query:
            matches = self.interest_rates_df[self.interest_rates_df['product_category'].str.contains("Education Loan", case=False)]
        elif "fixed deposit" in query or "fd" in query:
            matches = self.interest_rates_df[self.interest_rates_df['product_category'].str.contains("Fixed Deposit", case=False)]
        else:
            # Fallback to general search
            matches = self.interest_rates_df[self.interest_rates_df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
        
        if matches.empty:
            return f"No specific rate information found in the schedule for your query."
            
        # Return a clean summary of relevant columns
        cols = ['product_name', 'sub_type', 'interest_rate_pa', 'processing_fee', 'prepayment_charges']
        relevant_cols = [c for c in cols if c in matches.columns]
        return "Indicative Rates & Fees from Official Schedule:\n" + matches[relevant_cols].to_string(index=False)
        
    def lookup_loan_eligibility(self, query: str) -> str:
        if self.loan_eligibility_df is None:
            return "Loan eligibility data not available."
            
        query = query.lower()
        # Look for loan type in eligibility matrix
        if "home" in query:
            matches = self.loan_eligibility_df[self.loan_eligibility_df['loan_type'].str.contains("Home Loan", case=False)]
        elif "personal" in query:
            matches = self.loan_eligibility_df[self.loan_eligibility_df['loan_type'].str.contains("Personal Loan", case=False)]
        else:
            matches = self.loan_eligibility_df[self.loan_eligibility_df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]

        if matches.empty:
            return "No matching eligibility criteria found in the matrix."
            
        # Return top matches to avoid overwhelming context
        return "Indicative Eligibility Criteria (from Matrix):\n" + matches.head(3).to_string(index=False)
