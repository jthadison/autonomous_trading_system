"""
Paper Trading System Launcher (Direct API)
Central control script for managing the entire paper trading system
Updated to use Direct Oanda API instead of MCP server
"""

import sys
import os
import subprocess
import asyncio
import signal
from pathlib import Path
from datetime import datetime
import threading
import time

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

class PaperTradingController:
    """Central controller for the paper trading system"""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.dashboard_url = "http://localhost:8501"
        
    def check_prerequisites(self):
        """Check if all prerequisites are met for Direct API"""
        print("🔍 Checking prerequisites for Direct Oanda API...")
        
        issues = []
        
        # Check required Python packages for Direct API
        required_packages = [
            'oandapyV20',  # For Direct API
            'streamlit', 
            'pandas', 
            'plotly', 
            'crewai', 
            'langchain_anthropic', 
            'sqlalchemy'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                if package == 'oandapyV20':
                    print("✅ oandapyV20 library found")
            except ImportError:
                issues.append(f"Missing Python package: {package}")
        
        # Check environment variables for Direct API
        required_env = [
            'OANDA_API_KEY',
            'OANDA_ACCOUNT_ID', 
            'OANDA_ENVIRONMENT',
            'ANTHROPIC_API_KEY'
        ]
        
        for env_var in required_env:
            if not os.getenv(env_var):
                issues.append(f"Missing environment variable: {env_var}")
        
        # Test Direct Oanda API connection
        try:
            from src.mcp_servers.oanda_direct_api import OandaDirectAPI
            
            async def test_api():
                try:
                    async with OandaDirectAPI() as oanda:
                        account_info = await oanda.get_account_info()
                        return True
                except Exception as e:
                    print(f"Direct API connection error: {e}")
                    return False
            
            api_ok = asyncio.run(test_api())
            if api_ok:
                print("✅ Direct Oanda API connection successful")
            else:
                issues.append("Cannot connect to Oanda Direct API - check credentials")
        except Exception as e:
            issues.append(f"Cannot import/test Direct Oanda API: {e}")
        
        # Check LLM availability
        try:
            import litellm
            if os.getenv('ANTHROPIC_API_KEY'):
                print("✅ LLM (Anthropic) available")
            elif os.getenv('OPENAI_API_KEY'):
                print("✅ LLM (OpenAI) available")
            else:
                issues.append("No LLM API key found (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        except ImportError:
            pass  # litellm not required
        
        if issues:
            print("❌ Prerequisites check failed:")
            for issue in issues:
                print(f"   • {issue}")
            print("\n💡 Setup help:")
            print("   1. Install: pip install oandapyV20")
            print("   2. Set environment variables in .env:")
            print("      OANDA_API_KEY=your_token")
            print("      OANDA_ACCOUNT_ID=your_account_id")
            print("      OANDA_ENVIRONMENT=practice")
            print("      ANTHROPIC_API_KEY=your_anthropic_key")
            return False
        else:
            print("✅ All prerequisites met for Direct API!")
            return True
    
    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        print("🚀 Starting Paper Trading Dashboard...")
        
        # Use the updated dashboard file
        dashboard_script = project_root / "paper_trading_dashboard.py"
        
        # Verify the dashboard script exists
        if not dashboard_script.exists():
            print(f"❌ Dashboard script not found: {dashboard_script}")
            print("💡 Make sure paper_trading_dashboard.py is in the project root")
            return False
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_script),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        try:
            print(f"📂 Running dashboard from: {dashboard_script}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root)
            )
            self.processes['dashboard'] = process
            
            # Give it a moment to start
            time.sleep(3)
            
            if process.poll() is None:  # Process is still running
                print(f"✅ Dashboard started successfully!")
                print(f"🌐 Access dashboard at: {self.dashboard_url}")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Dashboard failed to start:")
                print(f"Stdout: {stdout.decode()}")
                print(f"Error: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return False
    
    def start_paper_trading_engine(self):
        """Start the paper trading engine with Direct API"""
        print("🤖 Starting Paper Trading Engine with Direct API...")
        
        try:
            # Import from root directory paper_trading_system.py
            from paper_trading_system import PaperTradingEngine
            
            async def run_engine():
                engine = PaperTradingEngine(initial_balance=100000.0)
                await engine.initialize()
                await engine.start_trading()
            
            # Run in a separate thread
            def engine_thread():
                try:
                    asyncio.run(run_engine())
                except Exception as e:
                    print(f"❌ Trading engine error: {e}")
            
            thread = threading.Thread(target=engine_thread, daemon=True)
            thread.start()
            self.processes['engine'] = thread
            
            print("✅ Paper Trading Engine started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start trading engine: {e}")
            return False
    
    def start_system(self, dashboard_only=False):
        """Start the complete paper trading system"""
        print("🚀 STARTING PAPER TRADING SYSTEM (Direct API)")
        print("=" * 50)
        
        if not self.check_prerequisites():
            return False
        
        success = True
        
        # Start dashboard
        if not self.start_dashboard():
            success = False
        
        # Start trading engine (unless dashboard only)
        if not dashboard_only:
            if not self.start_paper_trading_engine():
                success = False
        
        if success:
            self.running = True
            print("\n✅ Paper Trading System started successfully!")
            print("=" * 50)
            print("📊 Dashboard URL:", self.dashboard_url)
            print("🔗 Direct Oanda API: Connected")
            if not dashboard_only:
                print("🤖 Trading Engine: Running")
            else:
                print("🤖 Trading Engine: Dashboard Only Mode")
            print("⏹️  Press Ctrl+C to stop the system")
            print("=" * 50)
            return True
        else:
            print("\n❌ Failed to start some components")
            self.stop_system()
            return False
    
    def stop_system(self):
        """Stop all system components"""
        print("\n🛑 Stopping Paper Trading System...")
        
        # Stop dashboard
        if 'dashboard' in self.processes:
            try:
                self.processes['dashboard'].terminate()
                self.processes['dashboard'].wait(timeout=5)
                print("✅ Dashboard stopped")
            except Exception as e:
                print(f"⚠️ Error stopping dashboard: {e}")
                try:
                    self.processes['dashboard'].kill()
                except:
                    pass
        
        # Stop trading engine
        if 'engine' in self.processes:
            try:
                # The engine thread should stop when the main thread exits
                print("✅ Trading engine stopped")
            except Exception as e:
                print(f"⚠️ Error stopping trading engine: {e}")
        
        self.running = False
        print("📴 Paper Trading System stopped")
    
    def show_status(self):
        """Show current system status"""
        print("📊 PAPER TRADING SYSTEM STATUS")
        print("=" * 40)
        print(f"🕐 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 Dashboard URL: {self.dashboard_url}")
        print(f"⚡ System Running: {'Yes' if self.running else 'No'}")
        
        # Check Direct API connection
        try:
            from src.mcp_servers.oanda_direct_api import OandaDirectAPI
            
            async def check_api():
                try:
                    async with OandaDirectAPI() as oanda:
                        account_info = await oanda.get_account_info()
                        return True, account_info.get('currency', 'USD')
                except Exception as e:
                    return False, str(e)
            
            api_ok, info = asyncio.run(check_api())
            if api_ok:
                print(f"🔗 Direct Oanda API: ✅ Connected ({info})")
            else:
                print(f"🔗 Direct Oanda API: ❌ Disconnected ({info})")
        except Exception as e:
            print(f"🔗 Direct Oanda API: ❌ Error ({e})")
        
        # Check processes
        dashboard_running = 'dashboard' in self.processes and self.processes['dashboard'].poll() is None
        engine_running = 'engine' in self.processes and self.processes['engine'].is_alive()
        
        print(f"📊 Dashboard: {'✅ Running' if dashboard_running else '❌ Stopped'}")
        print(f"🤖 Trading Engine: {'✅ Running' if engine_running else '❌ Stopped'}")
        
        print("=" * 40)
    
    def wait_for_shutdown(self):
        """Wait for user to stop the system"""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n📴 Received shutdown signal")
            self.stop_system()

def main():
    """Main function"""
    controller = PaperTradingController()
    symbol = "EUR_USD"
    timeframe = "M5"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            dashboard_only = "--dashboard-only" in sys.argv
            if controller.start_system(dashboard_only=dashboard_only):
                controller.wait_for_shutdown()
        
        elif command == "dashboard":
            if controller.start_system(dashboard_only=True):
                controller.wait_for_shutdown()
        
        elif command == "stop":
            controller.stop_system()
        
        elif command == "status":
            controller.show_status()
        
        elif command == "check":
            controller.check_prerequisites()
        
        elif command == "setup":
            print("🛠️ SETUP GUIDE FOR DIRECT OANDA API")
            print("=" * 50)
            print()
            print("1. Get Oanda API Credentials:")
            print("   • Visit: https://www.oanda.com/account/login")
            print("   • Go to 'Manage API Access'")
            print("   • Generate API token")
            print("   • Copy your account ID")
            print()
            print("2. Create .env file in project root:")
            print("   OANDA_API_KEY=your_api_token_here")
            print("   OANDA_ACCOUNT_ID=your_account_id_here")
            print("   OANDA_ENVIRONMENT=practice")
            print("   ANTHROPIC_API_KEY=your_anthropic_key_here")
            print()
            print("3. Install required packages:")
            print("   pip install oandapyV20")
            print()
            print("4. Test setup:")
            print("   python paper_trading_launcher.py check")
            print()
            print("5. Start system:")
            print("   python paper_trading_launcher.py start")
            print()
        
        elif command == "test":
            print("🧪 TESTING DIRECT API CONNECTION")
            print("=" * 40)
            try:
                from src.mcp_servers.oanda_direct_api import OandaDirectAPI
                
                async def test_all():
                    print("Testing Direct API connection...")
                    async with OandaDirectAPI() as oanda:
                        # Test account info
                        account_info = await oanda.get_account_info()
                        print(f"✅ Account: {account_info.get('currency', 'USD')}")
                        
                        # Test price feed
                        price = await oanda.get_current_price(symbol)
                        print(f"✅ {symbol} Price: {price.get('bid', 'N/A')}")
                        
                        # Test historical data
                        historical = await oanda.get_historical_data(symbol, timeframe, 5)
                        print(f"✅ Historical Data: {len(historical.get('data', []))} candles")
                        
                        print("\n🎉 All Direct API tests passed!")
                
                asyncio.run(test_all())
                
            except Exception as e:
                print(f"❌ Direct API test failed: {e}")
        
        else:
            print_usage()
    
    else:
        # Interactive mode
        print("📈 PAPER TRADING SYSTEM CONTROLLER (Direct API)")
        print("=" * 60)
        print("1. Start complete system (engine + dashboard)")
        print("2. Start dashboard only")
        print("3. Check prerequisites")
        print("4. Show system status")
        print("5. Setup guide")
        print("6. Test Direct API")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            if controller.start_system(dashboard_only=False):
                controller.wait_for_shutdown()
        
        elif choice == "2":
            if controller.start_system(dashboard_only=True):
                controller.wait_for_shutdown()
        
        elif choice == "3":
            controller.check_prerequisites()
        
        elif choice == "4":
            controller.show_status()
        
        elif choice == "5":
            main()  # Call setup
            sys.argv = ["", "setup"]
            main()
        
        elif choice == "6":
            sys.argv = ["", "test"]
            main()
        
        elif choice == "7":
            print("👋 Goodbye!")
            
        else:
            print("Invalid choice")

def print_usage():
    """Print usage instructions"""
    print("\nUsage:")
    print("  python paper_trading_launcher.py start              # Start complete system")
    print("  python paper_trading_launcher.py start --dashboard-only  # Dashboard only")
    print("  python paper_trading_launcher.py dashboard         # Dashboard only")
    print("  python paper_trading_launcher.py stop              # Stop system")
    print("  python paper_trading_launcher.py status            # Show status")
    print("  python paper_trading_launcher.py check             # Check prerequisites")
    print("  python paper_trading_launcher.py setup             # Setup guide")
    print("  python paper_trading_launcher.py test              # Test Direct API")

# Only set up signal handlers if running as main script, not when imported by Streamlit
if __name__ == "__main__":
    # Check if we're running under Streamlit
    if 'streamlit' not in sys.modules and 'STREAMLIT_SERVER_PORT' not in os.environ:
        # Handle shutdown signals gracefully only if not in Streamlit
        def signal_handler(signum, frame):
            print("\n📴 Received shutdown signal")
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            # Signal handlers can only be set from main thread
            pass
    
    main()