from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your TensorFlow model
model = tf.keras.models.load_model("model")

# Load tokenizer, df_train, and df_severity
# Assuming these are loaded from files or are already available in your environment
# For example:
# tokenizer = ...
# df_train = pd.read_csv('...')
# df_severity = pd.read_csv('./data/severity_levels.csv')

# Prepare category reference dataframe
df_ref = (
    df_train[["category", "category_codes"]].drop_duplicates().reset_index(drop=True)
)

# FastAPI app
app = FastAPI()


# Request model
class RequestModel(BaseModel):
    queries: list


# Response model
class ResponseModel(BaseModel):
    predicted_category: str
    severity_level: str


@app.post("/predict", response_model=list[ResponseModel])
async def predict(request: RequestModel):
    try:
        # Convert queries to padded sequences
        input_text_sequences = tokenizer.texts_to_sequences(request.queries)
        input_text_padded = pad_sequences(
            input_text_sequences, maxlen=30, padding="post", truncating="post"
        )

        # Predict outcomes
        predictions = model.predict(input_text_padded)
        predicted_classes = np.argmax(predictions, axis=1)

        responses = []
        for pred_class in predicted_classes:
            category = df_ref[df_ref["category_codes"] == pred_class][
                "category"
            ].values[0]
            severity_level = df_severity[df_severity["category"] == category][
                "severity_level"
            ].values[0]

            responses.append(
                ResponseModel(
                    predicted_category=category, severity_level=severity_level
                )
            )

        return responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
