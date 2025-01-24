import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
from typing import List, Dict, Any
import logging

class GoogleSheetsManager:
    def __init__(self, credentials_path: str):
        """구글 시트 매니저 초기화"""
        try:
            # .env 파일 로드
            load_dotenv()
            self.spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID')
            if not self.spreadsheet_id:
                raise ValueError("GOOGLE_SHEETS_ID가 환경 변수에 설정되지 않았습니다.")
            
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.service = build('sheets', 'v4', credentials=self.credentials)
            print(f"구글 시트 서비스 초기화 성공: {credentials_path}")
        except Exception as e:
            logging.error(f"구글 시트 서비스 초기화 실패: {e}")
            raise

    def save_to_sheets(self, classified_df: pd.DataFrame, unclassified_df: pd.DataFrame) -> str:
        """분류된/미분류된 데이터프레임을 구글 시트에 저장"""
        try:
            # 1. 시트 설정 확인
            self._setup_sheets()
            
            # 2. 기존 데이터 초기화
            batch_clear_request = {
                'requests': [
                    {
                        'updateCells': {
                            'range': {
                                'sheetId': self._get_sheet_id('Classified'),
                                'startRowIndex': 0,
                                'startColumnIndex': 0
                            },
                            'fields': 'userEnteredValue'
                        }
                    },
                    {
                        'updateCells': {
                            'range': {
                                'sheetId': self._get_sheet_id('Unclassified'),
                                'startRowIndex': 0,
                                'startColumnIndex': 0
                            },
                            'fields': 'userEnteredValue'
                        }
                    }
                ]
            }
            
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body=batch_clear_request
            ).execute()
            print("기존 데이터 초기화 완료")
            
            # 3. 분류된 데이터 저장
            if not classified_df.empty:
                values = [classified_df.columns.values.tolist()] + classified_df.values.tolist()
                self.service.spreadsheets().values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range='Classified!A1',
                    valueInputOption='RAW',
                    body={'values': values}
                ).execute()
                print("분류된 데이터 저장 완료")

            # 4. 미분류 데이터 저장
            if not unclassified_df.empty:
                values = [unclassified_df.columns.values.tolist()] + unclassified_df.values.tolist()
                self.service.spreadsheets().values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range='Unclassified!A1',
                    valueInputOption='RAW',
                    body={'values': values}
                ).execute()
                print("미분류 데이터 저장 완료")

            # 5. 서식 설정
            self._format_sheets()
            
            return self.spreadsheet_id

        except Exception as e:
            logging.error(f"시트 저장 중 오류 발생: {e}")
            raise

    def _setup_sheets(self) -> None:
        """시트 초기 설정"""
        try:
            # 기존 시트 정보 가져오기
            spreadsheet = self.service.spreadsheets().get(spreadsheetId=self.spreadsheet_id).execute()
            existing_sheets = {sheet['properties']['title']: sheet['properties']['sheetId'] 
                             for sheet in spreadsheet.get('sheets', [])}
            
            requests = []
            
            # 필요한 시트가 없는 경우에만 생성
            required_sheets = ['Classified', 'Unclassified']
            for sheet_title in required_sheets:
                if sheet_title not in existing_sheets:
                    requests.append({
                        'addSheet': {
                            'properties': {
                                'title': sheet_title,
                                'gridProperties': {
                                    'frozenRowCount': 1
                                }
                            }
                        }
                    })
            
            # 요청이 있는 경우에만 실행
            if requests:
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={'requests': requests}
                ).execute()
                print("새 시트 생성 완료")
            
        except Exception as e:
            logging.error(f"시트 초기 설정 중 오류 발생: {e}")
            raise

    def _format_sheets(self) -> None:
        """시트 서식 설정"""
        try:
            spreadsheet = self.service.spreadsheets().get(spreadsheetId=self.spreadsheet_id).execute()
            
            requests = []
            for sheet in spreadsheet['sheets']:
                sheet_id = sheet['properties']['sheetId']
                requests.append({
                    'repeatCell': {
                        'range': {
                            'sheetId': sheet_id,
                            'startRowIndex': 0,
                            'endRowIndex': 1
                        },
                        'cell': {
                            'userEnteredFormat': {
                                'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8},
                                'textFormat': {'bold': True},
                                'horizontalAlignment': 'CENTER'
                            }
                        },
                        'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                    }
                })

            if requests:
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={'requests': requests}
                ).execute()
                print("시트 서식 설정 완료")
                
        except Exception as e:
            logging.error(f"시트 서식 설정 오류: {e}")
            raise

    def save_hierarchical_entities(self, sheet_id: str, hierarchy: Dict[str, Any]) -> bool:
        """계층적 엔터티 구조를 시트에 저장"""
        try:
            # 헤더 생성
            values = [["클러스터 엔터티", "그룹 엔터티", "청크 엔터티", "청크 텍스트"]]
            
            # 계층 구조를 행으로 변환
            for cluster_entity, group_data in hierarchy.items():
                for group_entity, chunk_data in group_data.items():
                    for chunk_text, chunk_entities in chunk_data.items():
                        # 청크 엔터티가 리스트인 경우 문자열로 변환
                        chunk_entities_str = ", ".join(chunk_entities) if isinstance(chunk_entities, list) else str(chunk_entities)
                        values.append([
                            cluster_entity,
                            group_entity,
                            chunk_entities_str,
                            chunk_text
                        ])

            # 데이터 업데이트
            body = {'values': values}
            self.service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range="A1",  # 항상 A1부터 시작
                valueInputOption='RAW',
                body=body
            ).execute()

            # 시트 서식 설정
            self._format_sheet(sheet_id)
            
            print(f"계층적 엔터티 저장 완료: https://docs.google.com/spreadsheets/d/{sheet_id}")
            return True

        except Exception as e:
            logging.error(f"계층적 엔터티 저장 오류: {e}")
            return False

    def _format_sheet(self, sheet_id: str) -> None:
        """시트 서식 설정"""
        try:
            requests = [{
                'updateSheetProperties': {
                    'properties': {
                        'gridProperties': {
                            'frozenRowCount': 1  # 헤더 행 고정
                        }
                    },
                    'fields': 'gridProperties.frozenRowCount'
                }
            }, {
                'repeatCell': {
                    'range': {'sheetId': 0, 'startRowIndex': 0, 'endRowIndex': 1},
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8},
                            'textFormat': {'bold': True}
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat)'
                }
            }]

            self.service.spreadsheets().batchUpdate(
                spreadsheetId=sheet_id,
                body={'requests': requests}
            ).execute()

        except Exception as e:
            logging.error(f"시트 서식 설정 오류: {e}") 

    def _get_sheet_id(self, sheet_title: str) -> int:
        """시트 ID 가져오기"""
        spreadsheet = self.service.spreadsheets().get(spreadsheetId=self.spreadsheet_id).execute()
        for sheet in spreadsheet.get('sheets', []):
            if sheet['properties']['title'] == sheet_title:
                return sheet['properties']['sheetId']
        raise ValueError(f"시트를 찾을 수 없음: {sheet_title}") 