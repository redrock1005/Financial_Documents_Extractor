from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import os

class GoogleSheetsManager:
    def __init__(self, credentials_path: str):
        self.SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        self.credentials_path = credentials_path
        self.credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=self.SCOPES)
        self.service = build('sheets', 'v4', credentials=self.credentials)
        
    def create_sheet(self, title: str) -> str:
        """새로운 구글 시트 생성"""
        spreadsheet = {
            'properties': {'title': title}
        }
        spreadsheet = self.service.spreadsheets().create(body=spreadsheet).execute()
        return spreadsheet['spreadsheetId']
    
    def update_values(self, spreadsheet_id: str, range_name: str, values: list):
        """시트 데이터 업데이트"""
        body = {'values': values}
        self.service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='RAW',
            body=body
        ).execute() 