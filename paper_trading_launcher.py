"""
Paper Trading System Launcher
Central control script for managing the entire paper trading system
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
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

class PaperTradingController:
    """Central controller for the paper trading system"""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.dashboard_url = "http://localhost:8501"
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("🔍 Checking prerequisites...")
        
        issues = []
        
        # Check environment variables
        required_env = ['ANTHROPIC_API_KEY']  # Add others as needed
        for env_var in required_env:
            if not os.getenv(env_var):
                issues.append(f"Missing environment variable: {env_var}")
        
        # Check if Oanda MCP server is running
        try:
            import aiohttp
            async def check_oanda():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:8000") as response:
                            return response.status == 200
                except:
                    return False
            
            oanda_ok = asyncio.run(check_oanda())
            if not oanda_ok:
                issues.append("Oanda MCP server not running on localhost:8000")
        except Exception as e:
            issues.append(f"Cannot connect to Oanda MCP server: {e}")
        
        # Check required Python packages
        required_packages = [
            'streamlit', 'pandas', 'plotly', 'crewai', 
            'langchain_anthropic', 'aiohttp', 'sqlalchemy'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Missing Python package: {package}")
        
        if issues:
            print("❌ Prerequisites check failed:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ All prerequisites met!")
            return True
    
    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        print("🚀 Starting Paper Trading Dashboard...")
        
        # Try multiple possible locations for the dashboard script
        possible_locations = [
            project_root / "paper_trading_dashboard.py",  # Project root
            project_root / "src" / "dashboard" / "paper_trading_dashboard.py",  # src/dashboard
            Path(__file__).parent / "paper_trading_dashboard.py",  # Same directory as launcher
        ]
        
        dashboard_script = None
        for location in possible_locations:
            if location.exists():
                dashboard_script = location
                break
        
        if dashboard_script is None:
            print(f"❌ Dashboard script not found in any of these locations:")
            for loc in possible_locations:
                print(f"   • {loc}")
            print("💡 Make sure paper_trading_dashboard.py exists in one of these locations")
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
        """Start the paper trading engine"""
        print("🤖 Starting Paper Trading Engine...")
        
        try:
            # Try to import from different possible locations
            try:
                from paper_trading_system import PaperTradingEngine
            except ImportError:
                try:
                    from src.dashboard.paper_trading_system import PaperTradingEngine
                except ImportError:
                    # Add the directory containing the paper trading system to path
                    possible_paths = [
                        project_root,
                        project_root / "src" / "dashboard",
                        Path(__file__).parent
                    ]
                    
                    for path in possible_paths:
                        sys.path.insert(0, str(path))
                        try:
                            from paper_trading_system import PaperTradingEngine
                            break
                        except ImportError:
                            continue
                    else:
                        raise ImportError("Could not find paper_trading_system module")
            
            async def run_engine():
                engine = PaperTradingEngine(initial_balance=100000.0)
                await engine.initialize()
                await engine.start_trading()
            
            # Run in a separate thread
            def engine_thread():
                asyncio.run(run_engine())
            
            thread = threading.Thread(target=engine_thread, daemon=True)
            thread.start()
            self.processes['engine'] = thread
            
            print("✅ Paper Trading Engine started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start trading engine: {e}")
            print(f"   Error details: {type(e).__name__}: {str(e)}")
            return False
    
    def start_system(self, dashboard_only=False):
        """Start the complete paper trading system"""
        print("🚀 STARTING PAPER TRADING SYSTEM")
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
            if not dashboard_only:
                print("🤖 Trading Engine: Running")
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
        """Show system status"""
        print("📊 PAPER TRADING SYSTEM STATUS")
        print("=" * 40)
        
        # Check dashboard
        if 'dashboard' in self.processes:
            if self.processes['dashboard'].poll() is None:
                print("📈 Dashboard: ✅ Running")
                print(f"   URL: {self.dashboard_url}")
            else:
                print("📈 Dashboard: ❌ Stopped")
        else:
            print("📈 Dashboard: ❌ Not Started")
        
        # Check trading engine
        if 'engine' in self.processes:
            if self.processes['engine'].is_alive():
                print("🤖 Trading Engine: ✅ Running")
            else:
                print("🤖 Trading Engine: ❌ Stopped")
        else:
            print("🤖 Trading Engine: ❌ Not Started")
        
        # Check external dependencies
        try:
            import aiohttp
            async def check_oanda():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:8000") as response:
                            return response.status == 200
                except:
                    return False
            
            oanda_ok = asyncio.run(check_oanda())
            print(f"🔗 Oanda MCP: {'✅ Connected' if oanda_ok else '❌ Disconnected'}")
        except:
            print("🔗 Oanda MCP: ❌ Disconnected")
        
        print("=" * 40)
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n📴 Shutdown signal received...")
            self.stop_system()

def main():
    """Main function with CLI interface"""
    
    controller = PaperTradingController()
    
    print("📈 PAPER TRADING SYSTEM CONTROLLER")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            dashboard_only = "--dashboard-only" in sys.argv
            if controller.start_system(dashboard_only=dashboard_only):
                controller.wait_for_shutdown()
        
        elif command == "stop":
            controller.stop_system()
        
        elif command == "status":
            controller.show_status()
        
        elif command == "dashboard":
            if controller.start_system(dashboard_only=True):
                controller.wait_for_shutdown()
        
        elif command == "check":
            controller.check_prerequisites()
        
        else:
            print(f"Unknown command: {command}")
            print_usage()
    
    else:
        # Interactive mode
        print("Choose an option:")
        print("1. Start complete system (engine + dashboard)")
        print("2. Start dashboard only")
        print("3. Check prerequisites")
        print("4. Show system status")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
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

# FIX: Only set up signal handlers if running as main script, not when imported by Streamlit
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