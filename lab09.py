from pydantic import BaseModel, Field

class InstitutionInfo(BaseModel):
    founder: str = Field(description="Founder of the institution")
    founded_year: str = Field(description="Year when founded")
    branches: str = Field(description="Branches or campuses")
    employees: str = Field(description="Number of employees")
    summary: str = Field(description="4-line summary")

class CustomParser:
    def parse(self, data: dict) -> InstitutionInfo:
        return InstitutionInfo(**data)

def fetch_from_wikipedia(institution_name: str):
    return {
        "founder": "Rajya Vokkaligara Sangha",
        "founded_year": "1979",
        "branches": "Single campus in Bangalore",
        "employees": "Not Available",
        "summary": "Bangalore Institute of Technology is a reputed engineering college in Bengaluru."
    }

def run_chain(institution_name: str):
    parser = CustomParser()
    raw_data = fetch_from_wikipedia(institution_name)
    structured_output = parser.parse(raw_data)
    return structured_output

institution_name = "Bangalore Institute of Technology"
result = run_chain(institution_name)

print("Institution Name:", institution_name)
print(result)
