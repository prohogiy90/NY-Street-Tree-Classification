from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from data_preparation import dataTransformer
from model import MLPModel

# Создание FastAPI-приложения
app = FastAPI()


# Определение Pydantic-модели
class Form(BaseModel):
    tree_id: int
    block_id: int
    created_at: str
    tree_dbh: int
    stump_diam: int
    curb_loc: str
    status: str
    spc_latin: str
    spc_common: str
    steward: str
    guards: str
    sidewalk: str
    user_type: str
    problems: str
    root_stone: str
    root_grate: str
    root_other: str
    trunk_wire: str
    trnk_light: str
    trnk_other: str
    brch_light: str
    brch_shoe: str
    brch_other: str
    address: str
    postcode: int
    zip_city: str
    community_board: int = Field(..., alias="community board")
    borocode: int
    borough: str
    cncldist: int
    st_assem: int
    st_senate: int
    nta: str
    nta_name: str
    boro_ct: int
    state: str
    latitude: float
    longitude: float
    x_sp: float
    y_sp: float
    council_district: float = Field(..., alias="council district")
    census_tract: float = Field(..., alias="census tract")
    bin: float
    bbl: float


# Эндпоинт для прогноза модели о состоянии дерева
@app.get("/tree_health_prediction/")
def tree_health_prediction(features: Form) -> str:
    """
    Принимает на вход признаки дерева в виде Pydantic-модели,
    выполняет предобработку данных и возвращает прогноз о состоянии дерева.
    """
    data_transformer = dataTransformer()
    input_data = data_transformer.transform(features)
    model = MLPModel()
    prediction = model.predict(input_data)
    return JSONResponse(content={"predicted_health": prediction})


if __name__ == "__main__":
    uvicorn.run(app, host='localhost')
