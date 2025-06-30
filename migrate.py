#!/usr/bin/env python3
"""
Automated Migration Script: MCP Server → Direct Oanda API
Helps automate the transition from MCP server to direct API integration
"""

import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any
import subprocess

class MigrationTool:
    """Automated migration helper"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.backup_suffix = ".mcp_backup"
        self.files_updated = []
        self.errors = []
    
    def check_prerequisites(self) -> bool:
        """Check if migration prerequisites are met"""
        print("🔍 Checking migration prerequisites...")
        
        issues = []
        
        # Check if oandapyV20 is installed
        try:
            import oandapyV20
            print("✅ oandapyV20 already installed")
        except ImportError:
            print("📦 Installing oandapyV20...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "oandapyV20"], 
                             check=True, capture_output=True)
                print("✅ oandapyV20 installed successfully")
            except subprocess.CalledProcessError as e:
                issues.append(f"Failed to install oandapyV20: {e}")
        
        # Check for required environment variables
        required_env_vars = [
            'OANDA_API_KEY',
            'OANDA_ACCOUNT_ID',
            'OANDA_ENVIRONMENT'
        ]
        
        missing_env_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_env_vars.append(var)
        
        if missing_env_vars:
            issues.append(f"Missing environment variables: {', '.join(missing_env_vars)}")
            print("⚠️ Missing environment variables. Please add to your .env file:")
            for var in missing_env_vars:
                print(f"   {var}=your_value_here")
        else:
            print("✅ Required environment variables found")
        
        # Check if direct API file exists
        direct_api_path = self.project_root / "src/mcp_servers/oanda_direct_api.py"
        if not direct_api_path.exists():
            issues.append("Direct API wrapper not found. Please create oanda_direct_api.py first.")
        else:
            print("✅ Direct API wrapper found")
        
        if issues:
            print("\n❌ Prerequisites not met:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        
        print("✅ All prerequisites met!")
        return True
    
    def backup_file(self, file_path: Path) -> bool:
        """Create backup of file before modification"""
        try:
            backup_path = file_path.with_suffix(file_path.suffix + self.backup_suffix)
            shutil.copy2(file_path, backup_path)
            print(f"📄 Backed up: {file_path.name}")
            return True
        except Exception as e:
            self.errors.append(f"Failed to backup {file_path}: {e}")
            return False
    
    def update_imports(self, file_path: Path) -> bool:
        """Update MCP imports to direct API imports"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace import statements
            replacements = [
                (
                    r'from src\.mcp_servers\.oanda_mcp_wrapper import OandaMCPWrapper',
                    'from src.mcp_servers.oanda_direct_api import OandaDirectAPI'
                ),
                (
                    r'from \.\.mcp_servers\.oanda_mcp_wrapper import OandaMCPWrapper',
                    'from ..mcp_servers.oanda_direct_api import OandaDirectAPI'
                ),
                (
                    r'OandaMCPWrapper\("http://localhost:8000"\)',
                    'OandaDirectAPI()'
                ),
                (
                    r'OandaMCPWrapper\(\)',
                    'OandaDirectAPI()'
                )
            ]
            
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.files_updated.append(str(file_path))
                print(f"✏️ Updated: {file_path.name}")
                return True
            else:
                print(f"⏭️ No changes needed: {file_path.name}")
                return False
            
        except Exception as e:
            self.errors.append(f"Failed to update {file_path}: {e}")
            return False
    
    def find_files_to_update(self) -> List[Path]:
        """Find all Python files that import MCP wrapper"""
        files_to_update = []
        
        # Search patterns
        patterns = [
            "src/**/*.py",
            "*.py"
        ]
        
        for pattern in patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file() and file_path.suffix == '.py':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check if file imports MCP wrapper
                        if 'oanda_mcp_wrapper' in content or 'OandaMCPWrapper' in content:
                            files_to_update.append(file_path)
                    except Exception:
                        continue  # Skip files that can't be read
        
        return files_to_update
    
    def update_environment_template(self) -> bool:
        """Update .env template with Oanda API variables"""
        env_template_path = self.project_root / ".env.template"
        env_example_path = self.project_root / ".env.example"
        
        template_content = """
# Oanda API Configuration (REQUIRED for direct API)
OANDA_ACCESS_TOKEN=your_oanda_access_token_here
OANDA_ACCOUNT_ID=your_oanda_account_id_here  
OANDA_ENVIRONMENT=practice  # or 'live' for real trading

# CrewAI Agents (REQUIRED)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional fallback
OPENAI_API_KEY=your_openai_api_key_here

# Database (Optional - defaults to SQLite)
DATABASE_URL=your_database_url_here

# Logging
LOG_LEVEL=INFO
"""
        
        # Update .env.template if it exists
        if env_template_path.exists():
            self.backup_file(env_template_path)
            with open(env_template_path, 'w') as f:
                f.write(template_content)
            print("✏️ Updated .env.template")
        
        # Create .env.example if it doesn't exist
        if not env_example_path.exists():
            with open(env_example_path, 'w') as f:
                f.write(template_content)
            print("📄 Created .env.example")
        
        return True
    
    def update_readme_files(self) -> bool:
        """Update README files to remove MCP server references"""
        readme_files = [
            self.project_root / "README.md",
            self.project_root / "src/dashboard/README_paper_trading.md"
        ]
        
        for readme_path in readme_files:
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Remove MCP server setup sections
                    mcp_patterns = [
                        r'### \*\*Step \d+: Start Oanda MCP Server\*\*.*?```\n',
                        r'## MCP Server Setup.*?(?=##)',
                        r'cd oanda-mcp-server.*?\n',
                        r'python server\.py.*?\n',
                        r'# Should run on http://localhost:8000.*?\n',
                        r'curl http://localhost:8000.*?\n',
                        r'BJLG-92.*?server.*?\n'
                    ]
                    
                    for pattern in mcp_patterns:
                        content = re.sub(pattern, '', content, flags=re.DOTALL)
                    
                    # Add direct API setup section
                    if "Direct Oanda API" not in content:
                        api_setup = """
### **Oanda API Setup**

Configure your Oanda credentials in `.env` file:
```bash
OANDA_ACCESS_TOKEN=your_token_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice
```

Get credentials from [OANDA](https://www.oanda.com/demo-account/).
"""
                        # Insert after environment variables section
                        content = re.sub(
                            r'(### \*\*Step \d+: Environment Variables\*\*.*?```)',
                            r'\1' + api_setup,
                            content,
                            flags=re.DOTALL
                        )
                    
                    if content != original_content:
                        self.backup_file(readme_path)
                        with open(readme_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        self.files_updated.append(str(readme_path))
                        print(f"✏️ Updated: {readme_path.name}")
                
                except Exception as e:
                    self.errors.append(f"Failed to update {readme_path}: {e}")
        
        return True
    
    def test_migration(self) -> bool:
        """Test the migration by running direct API test"""
        print("\n🧪 Testing migration...")
        
        try:
            # Test direct API import
            test_script = """
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

try:
    from src.mcp_servers.oanda_direct_api import OandaDirectAPI
    print("✅ Direct API import successful")
    
    import asyncio
    async def test():
        async with OandaDirectAPI() as oanda:
            health = await oanda.health_check()
            if health.get('status') == 'healthy':
                print("✅ Direct API connection successful")
                return True
            else:
                print(f"❌ API health check failed: {health.get('error', 'Unknown error')}")
                return False
    
    success = asyncio.run(test())
    if success:
        print("🎉 Migration test PASSED!")
    else:
        print("❌ Migration test FAILED!")
        
except Exception as e:
    print(f"❌ Migration test error: {e}")
    success = False
"""
            
            # Write temporary test script
            test_file = self.project_root / "temp_migration_test.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            # Run test
            result = subprocess.run([sys.executable, str(test_file)], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            # Clean up
            test_file.unlink()
            
            if result.returncode == 0:
                print("✅ Migration test passed!")
                print(result.stdout)
                return True
            else:
                print("❌ Migration test failed!")
                print(result.stderr)
                return False
                
        except Exception as e:
            self.errors.append(f"Migration test failed: {e}")
            return False
    
    def run_migration(self) -> bool:
        """Run the complete migration process"""
        print("🚀 Starting MCP → Direct API Migration")
        print("=" * 50)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            return False
        
        print("\n📁 Finding files to update...")
        files_to_update = self.find_files_to_update()
        
        if not files_to_update:
            print("✅ No files found that need updating")
        else:
            print(f"📋 Found {len(files_to_update)} files to update:")
            for file_path in files_to_update:
                print(f"   • {file_path.relative_to(self.project_root)}")
        
        # Step 2: Ask for confirmation
        if files_to_update:
            response = input(f"\n🤔 Update {len(files_to_update)} files? (y/N): ")
            if response.lower() != 'y':
                print("❌ Migration cancelled by user")
                return False
        
        # Step 3: Backup and update files
        print("\n📝 Updating files...")
        for file_path in files_to_update:
            self.backup_file(file_path)
            self.update_imports(file_path)
        
        # Step 4: Update environment template
        print("\n🔧 Updating environment templates...")
        self.update_environment_template()
        
        # Step 5: Update documentation
        print("\n📚 Updating documentation...")
        self.update_readme_files()
        
        # Step 6: Test migration
        if os.getenv('OANDA_ACCESS_TOKEN'):
            self.test_migration()
        else:
            print("\n⚠️ Skipping migration test - OANDA_ACCESS_TOKEN not set")
            print("   Set your Oanda credentials and run the test manually")
        
        # Step 7: Summary
        print("\n🎉 Migration Summary")
        print("=" * 30)
        print(f"✅ Files updated: {len(self.files_updated)}")
        
        if self.files_updated:
            print("\nUpdated files:")
            for file_path in self.files_updated:
                print(f"   • {file_path}")
        
        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   • {error}")
        
        print("\n🔄 Next Steps:")
        print("   1. Set your Oanda API credentials in .env file")
        print("   2. Test the direct API: python src/mcp_servers/oanda_direct_api.py")
        print("   3. Test your trading system: python src/dashboard/paper_trading_launcher.py")
        print("   4. Remove MCP server dependency when satisfied")
        
        print(f"\n💾 Backup files created with '{self.backup_suffix}' suffix")
        print("   Remove them when migration is confirmed working")
        
        return len(self.errors) == 0
    
    def rollback_migration(self) -> bool:
        """Rollback migration by restoring backup files"""
        print("🔄 Rolling back migration...")
        
        backup_files = list(self.project_root.rglob(f"*{self.backup_suffix}"))
        
        if not backup_files:
            print("❌ No backup files found")
            return False
        
        print(f"📋 Found {len(backup_files)} backup files to restore")
        
        response = input("🤔 Restore all backup files? (y/N): ")
        if response.lower() != 'y':
            print("❌ Rollback cancelled")
            return False
        
        restored = 0
        for backup_path in backup_files:
            try:
                original_path = backup_path.with_suffix(
                    backup_path.suffix.replace(self.backup_suffix, '')
                )
                shutil.copy2(backup_path, original_path)
                backup_path.unlink()  # Remove backup file
                restored += 1
                print(f"✅ Restored: {original_path.name}")
            except Exception as e:
                print(f"❌ Failed to restore {backup_path}: {e}")
        
        print(f"\n🎉 Rollback complete! Restored {restored} files")
        return True


def main():
    """Main migration function"""
    migration_tool = MigrationTool()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "rollback":
            migration_tool.rollback_migration()
            return
        elif sys.argv[1] == "check":
            migration_tool.check_prerequisites()
            return
    
    print("🔄 MCP Server → Direct Oanda API Migration Tool")
    print("=" * 60)
    print("This tool will help you migrate from MCP server to direct API")
    print()
    print("Options:")
    print("  python migrate.py         - Run full migration")
    print("  python migrate.py check   - Check prerequisites only")
    print("  python migrate.py rollback - Restore backup files")
    print()
    
    if len(sys.argv) == 1:
        response = input("🚀 Start migration? (y/N): ")
        if response.lower() == 'y':
            migration_tool.run_migration()
        else:
            print("❌ Migration cancelled")


if __name__ == "__main__":
    main()