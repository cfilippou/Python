# =========================
# Imports
# =========================

# Data handling libraries
import pandas as pd
import numpy as np

# Used to load the trained ML model
import joblib as jl

# PDF generation library
from fpdf import FPDF

# Used to add the current date to the PDF
from datetime import datetime

# FastAPI core imports
from fastapi import FastAPI, Request, Form

# Used to return files (PDF) as HTTP responses
from fastapi.responses import FileResponse

# Template engine for rendering HTML pages
from fastapi.templating import Jinja2Templates

# Preprocessing tools used before model prediction
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


# =========================
# Application setup
# =========================

# Load the trained machine learning model from disk
# This model is expected to already be trained and serialized
model = jl.load('model.joblib')

# Create FastAPI application instance
app = FastAPI()

# Configure Jinja2 to load templates from the "template" directory
template = Jinja2Templates(directory='template')


# =========================
# Routes
# =========================

@app.get('/')
def Home(request: Request):
    """
    Home route.
    Renders the index.html template and passes the request
    and page title to the template context.
    """
    return template.TemplateResponse(
        'index.html',
        {'request': request, 'title': 'Covid'}
    )


# =========================
# Global variables
# =========================

# Global DataFrame used to store uploaded data and predictions
# WARNING: Using global variables in FastAPI is NOT thread-safe
# and can cause issues with multiple users.
df = ''


@app.post('/predict')
def predict(file: str = Form(...)):
    """
    Accepts a CSV file path from an HTML form,
    preprocesses the data, runs predictions,
    and returns an HTML table with results.
    """
    global df  # Access the global dataframe

    # Read CSV file into a pandas DataFrame
    # NOTE: 'file' is assumed to be a path, not an uploaded file object
    df = pd.read_csv(file)

    # Select only the columns required by the ML model
    final = df[
        [
            'Age', 'Gender', 'COVID_Strain', 'Symptoms', 'Severity',
            'Hospitalized', 'ICU_Admission', 'Ventilator_Support',
            'Reinfection', 'Vaccination_Status', 'Doses_Received',
            'Occupation', 'Smoking_Status', 'BMI'
        ]
    ]

    # Initialize preprocessing tools
    # OrdinalEncoder converts categorical values to numbers
    # MinMaxScaler scales values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    oe = OrdinalEncoder()

    # Apply encoding first, then scaling
    final = pd.DataFrame(
        data=scaler.fit_transform(oe.fit_transform(final)),
        columns=final.columns
    )

    # Run predictions using the trained model
    pred = model.predict(final)

    # Build an HTML table as a string to return to frontend
    return_value = (
        '<table>'
        '<tr>'
        '<th>Age</th>'
        '<th>Gender</th>'
        '<th>COVID Strain</th>'
        '<th>Symptoms</th>'
        '<th>Severity</th>'
        '<th>Hospitalized</th>'
        '<th>ICU Admission</th>'
        '<th>Ventilator Support</th>'
        '<th>Reinfection</th>'
        '<th>Vaccination Status</th>'
        '<th>Doses Received</th>'
        '<th>Occupation</th>'
        '<th>Smoking Status</th>'
        '<th>BMI</th>'
        '<th>Predictions</th>'
        '</tr>'
    )

    # Create a new column for storing prediction labels
    df['predictions'] = ''

    # Loop through predictions and rows together
    for idx, value in enumerate(pred):
        # Convert numeric prediction to human-readable label
        df.loc[idx, 'predictions'] = (
            'Affected' if value == 1 else 'Not Affected'
        )

        # Append each row to the HTML table
        return_value += (
            '<tr>'
            f'<td>{df.loc[idx, "Age"]}</td>'
            f'<td>{df.loc[idx, "Gender"]}</td>'
            f'<td>{df.loc[idx, "COVID_Strain"]}</td>'
            f'<td>{df.loc[idx, "Symptoms"]}</td>'
            f'<td>{df.loc[idx, "Severity"]}</td>'
            f'<td>{df.loc[idx, "Hospitalized"]}</td>'
            f'<td>{df.loc[idx, "ICU_Admission"]}</td>'
            f'<td>{df.loc[idx, "Ventilator_Support"]}</td>'
            f'<td>{df.loc[idx, "Reinfection"]}</td>'
            f'<td>{df.loc[idx, "Vaccination_Status"]}</td>'
            f'<td>{df.loc[idx, "Doses_Received"]}</td>'
            f'<td>{df.loc[idx, "Occupation"]}</td>'
            f'<td>{df.loc[idx, "Smoking_Status"]}</td>'
            f'<td>{df.loc[idx, "BMI"]}</td>'
            f'<td>{df.loc[idx, "predictions"]}</td>'
            '</tr>'
        )

    # Close the HTML table
    return_value += "</table>"

    # Return table as JSON response
    return {'table': return_value}


@app.post('/exportPdf')
def exportPDF():
    """
    Generates a PDF report from the global dataframe
    and returns it as a downloadable file.
    """
    global df

    # Create a PDF in landscape mode
    pdf = FPDF(orientation='L', format='A4', unit='mm')
    pdf.add_page()

    # Title section
    pdf.set_font("Arial", size=18, style='B')
    pdf.cell(0, 10, 'Covid Predictions', align='C', ln=True)
    pdf.ln(10)

    # Table header styling
    pdf.set_font("Arial", size=12, style='B')
    pdf.set_fill_color(235, 235, 235)

    # First table headers
    pdf.cell(20, 10, 'Age', align='C', fill=True, border='R')
    pdf.cell(20, 10, 'Gender', align='C', fill=True, border='LR')
    pdf.cell(30, 10, 'COVID Strain', align='C', fill=True, border='LR')
    pdf.cell(30, 10, 'Symptoms', align='C', fill=True, border='LR')
    pdf.cell(25, 10, 'Severity', align='C', fill=True, border='LR')
    pdf.cell(30, 10, 'Hospitalized', align='C', fill=True, border='LR')
    pdf.cell(35, 10, 'ICU Admission', align='C', fill=True, border='LR')
    pdf.cell(45, 10, 'Ventilator Support', align='C', fill=True, border='LR')
    pdf.cell(30, 10, 'Reinfection', align='C', fill=True, border='L')
    pdf.ln(10)

    # Table body
    pdf.set_font("Arial", size=10)
    for _, row in df.iterrows():
        pdf.cell(20, 10, str(row['Age']), align='C', border='R')
        pdf.cell(20, 10, str(row['Gender']), align='C', border='R')
        pdf.cell(30, 10, str(row['COVID_Strain']), align='C', border='R')
        pdf.cell(30, 10, str(row['Symptoms']), align='C', border='R')
        pdf.cell(25, 10, str(row['Severity']), align='C', border='R')
        pdf.cell(30, 10, str(row['Hospitalized']), align='C', border='R')
        pdf.cell(35, 10, str(row['ICU_Admission']), align='C', border='R')
        pdf.cell(45, 10, str(row['Ventilator_Support']), align='C', border='R')
        pdf.cell(30, 10, str(row['Reinfection']), align='C')
        pdf.ln(10)

    # Second page for remaining columns
    pdf.add_page()

    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(45, 10, 'Vaccination Status', align='C', fill=True, border='R')
    pdf.cell(40, 10, 'Doses Received', align='C', fill=True, border='LR')
    pdf.cell(30, 10, 'Occupation', align='C', fill=True, border='LR')
    pdf.cell(40, 10, 'Smoking Status', align='C', fill=True, border='LR')
    pdf.cell(20, 10, 'BMI', align='C', fill=True, border='LR')
    pdf.cell(30, 10, 'Predictions', align='C', fill=True, border='L')
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    for _, row in df.iterrows():
        pdf.cell(45, 10, str(row['Vaccination_Status']), align='C', border='R')
        pdf.cell(40, 10, str(row['Doses_Received']), align='C', border='R')
        pdf.cell(30, 10, str(row['Occupation']), align='C', border='R')
        pdf.cell(40, 10, str(row['Smoking_Status']), align='C', border='R')
        pdf.cell(20, 10, str(row['BMI']), align='C', border='R')
        pdf.cell(30, 10, str(row['predictions']), align='C', border='L')
        pdf.ln(10)

    # Footer section
    pdf.ln(20)
    pdf.cell(170, 7, '')
    pdf.cell(40, 7, 'Dr. Christos Filippou', ln=True, align='C')
    pdf.cell(170, 7, '')
    pdf.cell(40, 7, 'Kardioxirourgos', align='C')
    pdf.ln(30)
    pdf.cell(170, 7, '')
    pdf.cell(40, 7, datetime.now().strftime('%d/%m/%Y'), align='C')

    # Save and return PDF file
    filename = 'covid.pdf'
    pdf.output(filename)

    return FileResponse(
        filename=filename,
        media_type="application/pdf",
        path=filename
    )