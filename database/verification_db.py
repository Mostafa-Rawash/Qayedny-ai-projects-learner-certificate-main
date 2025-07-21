import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

class VerificationDatabase:
    """Database manager for verification results"""
    
    def __init__(self, db_path: str = "verification_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database with necessary tables"""
        try:
            # Check if database already exists
            db_exists = os.path.exists(self.db_path)
            if db_exists:
                self.logger.info(f"Database already exists at {self.db_path}, preserving existing data")
            else:
                self.logger.info(f"Creating new database at {self.db_path}")
            
            # Connect to database (creates it if it doesn't exist)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create users table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_name TEXT NOT NULL UNIQUE,
                        total_hours REAL DEFAULT 0
                    )
                """)
                
                # Create fields table - must be created before submissions since it's referenced
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fields (
                        field_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        field_name TEXT NOT NULL UNIQUE,
                        total_hours REAL DEFAULT 0
                    )
                """)
                
                # Create submissions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS submissions (
                        submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        category TEXT NOT NULL,
                        activity_title TEXT,
                        verification_status TEXT DEFAULT 'pending',
                        original_time REAL,  -- Time in hours
                        calculated_time REAL,  -- Time in hours after verification
                        time_unit TEXT DEFAULT 'hours',  -- 'hours' or 'months'
                        multiplier REAL DEFAULT 1,
                        verification_report TEXT,
                        confidence_score REAL,
                        recommendations TEXT,
                        image_path TEXT,
                        field_id INTEGER,  -- Foreign key to fields table
                        FOREIGN KEY (user_id) REFERENCES users(user_id),
                        FOREIGN KEY (field_id) REFERENCES fields(field_id)
                    )
                """)
                
                # Drop existing triggers if they exist
                cursor.execute("DROP TRIGGER IF EXISTS update_user_hours_insert")
                cursor.execute("DROP TRIGGER IF EXISTS update_user_hours_update")
                cursor.execute("DROP TRIGGER IF EXISTS update_field_hours_insert")
                cursor.execute("DROP TRIGGER IF EXISTS update_field_hours_update")
                
                # Create trigger for INSERT operations on user hours
                cursor.execute("""
                    CREATE TRIGGER update_user_hours_insert
                    AFTER INSERT ON submissions
                    BEGIN
                        UPDATE users 
                        SET total_hours = (
                            SELECT ROUND(CAST(SUM(calculated_time) AS FLOAT), 2)
                            FROM submissions 
                            WHERE user_id = NEW.user_id
                            AND verification_status IN ('verified', 'submitted')
                        )
                        WHERE user_id = NEW.user_id;
                    END;
                """)
                
                # Create trigger for UPDATE operations on user hours
                cursor.execute("""
                    CREATE TRIGGER update_user_hours_update
                    AFTER UPDATE ON submissions
                    WHEN NEW.verification_status != OLD.verification_status 
                         OR NEW.calculated_time != OLD.calculated_time
                    BEGIN
                        UPDATE users 
                        SET total_hours = (
                            SELECT ROUND(CAST(SUM(calculated_time) AS FLOAT), 2)
                            FROM submissions 
                            WHERE user_id = NEW.user_id
                            AND verification_status IN ('verified', 'submitted')
                        )
                        WHERE user_id = NEW.user_id;
                    END;
                """)
                
                # Create trigger for INSERT operations on field hours
                cursor.execute("""
                    CREATE TRIGGER update_field_hours_insert
                    AFTER INSERT ON submissions
                    WHEN NEW.field_id IS NOT NULL
                    BEGIN
                        UPDATE fields 
                        SET total_hours = (
                            SELECT ROUND(CAST(SUM(calculated_time) AS FLOAT), 2)
                            FROM submissions 
                            WHERE field_id = NEW.field_id
                            AND verification_status IN ('verified', 'submitted')
                        )
                        WHERE field_id = NEW.field_id;
                    END;
                """)
                
                # Create trigger for UPDATE operations on field hours
                cursor.execute("""
                    CREATE TRIGGER update_field_hours_update
                    AFTER UPDATE ON submissions
                    WHEN NEW.field_id IS NOT NULL
                    AND (NEW.verification_status != OLD.verification_status 
                         OR NEW.calculated_time != OLD.calculated_time
                         OR NEW.field_id != OLD.field_id)
                    BEGIN
                        -- Update old field total if it exists
                        UPDATE fields 
                        SET total_hours = (
                            SELECT ROUND(CAST(SUM(calculated_time) AS FLOAT), 2)
                            FROM submissions 
                            WHERE field_id = OLD.field_id
                            AND verification_status IN ('verified', 'submitted')
                        )
                        WHERE field_id = OLD.field_id;
                        
                        -- Update new field total
                        UPDATE fields 
                        SET total_hours = (
                            SELECT ROUND(CAST(SUM(calculated_time) AS FLOAT), 2)
                            FROM submissions 
                            WHERE field_id = NEW.field_id
                            AND verification_status IN ('verified', 'submitted')
                        )
                        WHERE field_id = NEW.field_id;
                    END;
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise
            
    def create_user(self, user_name: str) -> int:
        """Create a new user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (user_name) VALUES (?)",
                    (user_name,)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # User already exists, get their ID
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE user_name = ?", (user_name,))
            return cursor.fetchone()[0]
            
    def create_field(self, field_name: str) -> int:
        """Create a new field and return its ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO fields (field_name) VALUES (?)",
                    (field_name,)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Field already exists, get its ID
            cursor = conn.cursor()
            cursor.execute("SELECT field_id FROM fields WHERE field_name = ?", (field_name,))
            return cursor.fetchone()[0]
            
    def save_submission(self, submission_data: Dict) -> int:
        """Save a new submission"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Ensure user exists
                user_id = self.create_user(submission_data['user_name'])
                
                # If field name is provided, ensure field exists and get its ID
                field_id = submission_data.get('field_id')
                if field_id is None and 'field' in submission_data:
                    field_id = self.create_field(submission_data['field'])
                
                cursor.execute("""
                    INSERT INTO submissions (
                        user_id, category, activity_title, verification_status,
                        original_time, calculated_time, verification_report,
                        confidence_score, recommendations, image_path, field_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    submission_data['category'],
                    submission_data.get('activity_title', ''),
                    submission_data.get('verification_status', 'pending'),
                    submission_data.get('original_time', 0),
                    submission_data.get('calculated_time', 0),
                    json.dumps(submission_data.get('verification_report', {})),
                    submission_data.get('confidence_score', 0.0),
                    json.dumps(submission_data.get('recommendations', [])),
                    submission_data.get('image_path', ''),
                    field_id
                ))
                
                submission_id = cursor.lastrowid
                conn.commit()
                return submission_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Error saving submission: {str(e)}")
            raise
            
    def get_user_submissions(self, user_name: str) -> List[Dict]:
        """Get all submissions for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, u.user_name, u.total_hours
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE u.user_name = ?
                    ORDER BY s.created_at DESC
                """, (user_name,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting user submissions: {str(e)}")
            raise
            
    def get_user_stats(self, user_name: str) -> Dict:
        """Get user statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        u.total_hours,
                        COUNT(s.submission_id) as total_submissions,
                        SUM(CASE WHEN s.verification_status = 'verified' THEN 1 ELSE 0 END) as verified_submissions,
                        AVG(s.confidence_score) as avg_confidence
                    FROM users u
                    LEFT JOIN submissions s ON u.user_id = s.user_id
                    WHERE u.user_name = ?
                    GROUP BY u.user_id
                """, (user_name,))
                
                row = cursor.fetchone()
                if not row:
                    return {
                        'total_hours': 0,
                        'total_submissions': 0,
                        'verified_submissions': 0,
                        'avg_confidence': 0
                    }
                    
                return {
                    'total_hours': row[0] or 0,
                    'total_submissions': row[1] or 0,
                    'verified_submissions': row[2] or 0,
                    'avg_confidence': row[3] or 0
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting user stats: {str(e)}")
            raise
            
    def save_verification_result(self, result: Dict) -> int:
        """Save verification result to database"""
        try:
            self.logger.info(f"Attempting to save verification result to database at {self.db_path}")
            self.logger.info(f"Verification data: status={result.get('verification_status')}, "
                           f"category={result.get('category')}, user={result.get('user_name')}")
            
            # Convert verification status to is_verified boolean
            is_verified = result['verification_status'] == 'verified'
            
            # If field name is provided, ensure field exists and get its ID
            field_id = None
            if 'field' in result:
                field_id = self.create_field(result['field'])
            
            # Prepare submission data
            submission_data = {
                'user_name': result.get('user_name', ''),
                'category': result['category'],
                'activity_title': result.get('activity_title', ''),
                'verification_status': result['verification_status'],
                'original_time': result.get('original_time', 0),
                'calculated_time': result.get('calculated_time', 0),
                'verification_report': result.get('verification_report', {}),
                'confidence_score': result.get('confidence_score', 0.0),
                'recommendations': result.get('recommendations', []),
                'image_path': result.get('image_path', ''),
                'field_id': field_id
            }
            
            # Add field name if it exists (for save_submission method compatibility)
            if 'field' in result:
                submission_data['field'] = result['field']
            
            # Use the existing save_submission method
            submission_id = self.save_submission(submission_data)
            self.logger.info(f"Successfully saved verification result as submission with ID: {submission_id}")
            return submission_id
                
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error saving verification result: {str(e)}")
            self.logger.error(f"Database path: {self.db_path}")
            self.logger.error(f"Result data: {result}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error saving verification result: {str(e)}")
            self.logger.error(f"Database path: {self.db_path}")
            self.logger.error(f"Result data: {result}")
            raise
            
    def get_verification_by_id(self, verification_id: int) -> Optional[Dict]:
        """Retrieve verification result by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, u.user_name
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.submission_id = ?
                """, (verification_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                result = dict(row)
                result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                
                return result
                
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving verification result: {str(e)}")
            raise
            
    def get_verifications_by_user(self, user_name: str) -> List[Dict]:
        """Retrieve all verification results for a specific user"""
        return self.get_user_submissions(user_name)
            
    def get_verifications_by_category(self, category: str) -> List[Dict]:
        """Retrieve all verification results for a specific category"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, u.user_name
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.category = ?
                    ORDER BY s.created_at DESC
                """, (category,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving category verifications: {str(e)}")
            raise
            
    def get_verification_stats(self) -> Dict:
        """Get statistics about verifications in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total counts
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as verified,
                        AVG(confidence_score) as avg_confidence
                    FROM submissions
                """)
                
                row = cursor.fetchone()
                
                # Get category breakdown
                cursor.execute("""
                    SELECT category, COUNT(*) as count
                    FROM submissions
                    GROUP BY category
                    ORDER BY count DESC
                """)
                
                category_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    'total_verifications': row[0] or 0,
                    'verified_count': row[1] or 0,
                    'average_confidence': float(row[2]) if row[2] is not None else 0.0,
                    'category_breakdown': category_stats
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting verification stats: {str(e)}")
            raise
            
    def get_recent_verifications(self, limit: int = 10) -> List[Dict]:
        """Get most recent verifications"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, u.user_name
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    ORDER BY s.created_at DESC
                    LIMIT ?
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting recent verifications: {str(e)}")
            raise
            
    def search_verifications(self, 
                           user_name: Optional[str] = None,
                           category: Optional[str] = None,
                           status: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[Dict]:
        """Search verifications with various filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT s.*, u.user_name
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE 1=1
                """
                params = []
                
                if user_name:
                    query += " AND u.user_name LIKE ?"
                    params.append(f"%{user_name}%")
                
                if category:
                    query += " AND s.category = ?"
                    params.append(category)
                
                if status:
                    is_verified = status.lower() == 'verified'
                    query += " AND s.verification_status = ?"
                    params.append(status)
                
                if start_date:
                    query += " AND s.created_at >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND s.created_at <= ?"
                    params.append(end_date)
                
                query += " ORDER BY s.created_at DESC"
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Error searching verifications: {str(e)}")
            raise
            
    def delete_verification(self, verification_id: int) -> bool:
        """Delete a verification record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM submissions WHERE submission_id = ?", 
                             (verification_id,))
                
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    self.logger.info(f"Verification {verification_id} deleted successfully")
                else:
                    self.logger.warning(f"Verification {verification_id} not found")
                
                return deleted
                
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting verification: {str(e)}")
            raise
            
    def cleanup_old_verifications(self, days_to_keep: int = 30) -> int:
        """Clean up old verification records"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM submissions 
                    WHERE datetime(created_at) < datetime('now', ?)
                """, (f'-{days_to_keep} days',))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old verification records")
                return deleted_count
                
        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up old verifications: {str(e)}")
            raise
            
    def export_verifications_to_json(self, output_file: str) -> None:
        """Export all verifications to a JSON file"""
        try:
            verifications = self.get_recent_verifications(limit=1000000)  # Get all verifications
            
            with open(output_file, 'w') as f:
                json.dump({
                    'verifications': verifications,
                    'stats': self.get_verification_stats(),
                    'exported_at': datetime.now().isoformat()
                }, f, indent=2)
                
            self.logger.info(f"Exported {len(verifications)} verifications to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting verifications: {str(e)}")
            raise

    def update_user_total_hours(self, user_id: int) -> None:
        """Update user's total hours based on their verified and submitted activities"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate total hours from verified and submitted activities
                cursor.execute("""
                    UPDATE users 
                    SET total_hours = (
                        SELECT ROUND(CAST(SUM(calculated_time) AS FLOAT), 2)
                        FROM submissions 
                        WHERE user_id = ?
                        AND verification_status IN ('verified', 'submitted')
                    )
                    WHERE user_id = ?
                """, (user_id, user_id))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Error updating user total hours: {str(e)}")
            raise

    def update_verification_status(self, verification_id: int, status: str) -> None:
        """
        Update the status of a verification record.
        
        Args:
            verification_id (int): ID of the verification to update
            status (str): New status to set
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # First get the user_id for this verification
                cursor.execute("SELECT user_id FROM submissions WHERE submission_id = ?", (verification_id,))
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"No submission found with ID {verification_id}")
                user_id = result[0]
                
                # Update the verification status
                cursor.execute(
                    """
                    UPDATE submissions 
                    SET verification_status = ?
                    WHERE submission_id = ?
                    """,
                    (status, verification_id)
                )
                
                # Update the user's total hours
                self.update_user_total_hours(user_id)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating verification status: {str(e)}")
            raise

    def get_field_stats(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get statistics for all fields, optionally filtered by user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        f.field_id,
                        f.field_name,
                        f.total_hours as total_field_hours,
                        COUNT(DISTINCT s.user_id) as total_users,
                        COUNT(s.submission_id) as total_submissions
                    FROM fields f
                    LEFT JOIN submissions s ON f.field_id = s.field_id
                """
                
                params = []
                if user_id is not None:
                    query += " WHERE s.user_id = ? "
                    params.append(user_id)
                
                query += """
                    GROUP BY f.field_id, f.field_name
                    ORDER BY f.total_hours DESC
                """
                
                cursor.execute(query, params)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting field stats: {str(e)}")
            raise

    def get_user_field_stats(self, user_name: str) -> Dict:
        """
        Get user's total hours and breakdown by field.
        Returns a dictionary containing:
        - total_hours: Total hours across all fields
        - fields: List of fields with their hours
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get user's total hours and basic info
                cursor.execute("""
                    SELECT 
                        u.user_id,
                        u.user_name,
                        u.total_hours,
                        COUNT(s.submission_id) as total_submissions,
                        SUM(CASE WHEN s.verification_status = 'verified' THEN 1 ELSE 0 END) as verified_submissions
                    FROM users u
                    LEFT JOIN submissions s ON u.user_id = s.user_id
                    WHERE u.user_name = ?
                    GROUP BY u.user_id
                """, (user_name,))
                
                user_row = cursor.fetchone()
                if not user_row:
                    return {
                        'user_name': user_name,
                        'total_hours': 0,
                        'total_submissions': 0,
                        'verified_submissions': 0,
                        'fields': []
                    }
                
                user_data = dict(user_row)
                
                # Get hours breakdown by field
                cursor.execute("""
                    SELECT 
                        f.field_id,
                        f.field_name,
                        ROUND(CAST(SUM(s.calculated_time) AS FLOAT), 2) as field_hours,
                        COUNT(s.submission_id) as field_submissions,
                        SUM(CASE WHEN s.verification_status = 'verified' THEN 1 ELSE 0 END) as field_verified
                    FROM fields f
                    JOIN submissions s ON f.field_id = s.field_id
                    WHERE s.user_id = ?
                    AND s.verification_status IN ('verified', 'submitted')
                    GROUP BY f.field_id, f.field_name
                    ORDER BY field_hours DESC
                """, (user_data['user_id'],))
                
                fields = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'user_name': user_data['user_name'],
                    'total_hours': user_data['total_hours'] or 0,
                    'total_submissions': user_data['total_submissions'] or 0,
                    'verified_submissions': user_data['verified_submissions'] or 0,
                    'fields': fields
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting user field stats: {str(e)}")
            raise

    def get_user_category_stats(self, user_name: str) -> Dict:
        """
        Get user's total hours and breakdown by category.
        Returns a dictionary containing category breakdown with hours and activity counts.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get user ID first
                cursor.execute("SELECT user_id FROM users WHERE user_name = ?", (user_name,))
                user_row = cursor.fetchone()
                if not user_row:
                    return {}
                
                user_id = user_row['user_id']
                
                # Get hours breakdown by category
                cursor.execute("""
                    SELECT 
                        category,
                        ROUND(CAST(SUM(calculated_time) AS FLOAT), 2) as category_hours,
                        COUNT(submission_id) as category_submissions,
                        SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as category_verified
                    FROM submissions
                    WHERE user_id = ?
                    AND verification_status IN ('verified', 'submitted')
                    GROUP BY category
                    ORDER BY category_hours DESC
                """, (user_id,))
                
                categories = {}
                for row in cursor.fetchall():
                    categories[row['category']] = {
                        'hours': row['category_hours'],
                        'count': row['category_submissions'],
                        'verified': row['category_verified']
                    }
                
                return categories
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting user category stats: {str(e)}")
            raise