import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Try to import services with fallback
try:
    from services.satellite_service import SatelliteService
except:
    SatelliteService = None

try:
    from services.tle_service import TLEService
except:
    TLEService = None

try:
    from sgp4.api import Satrec, jday
except:
    Satrec = None
    jday = None

logger = logging.getLogger(__name__)

class VisualizationService:
    """Service for creating enhanced satellite trajectory visualizations"""
    
    def __init__(self):
        self.satellite_service = SatelliteService() if SatelliteService else None
        self.tle_service = TLEService() if TLEService else None
        
        # Set matplotlib style for better visuals
        plt.style.use('dark_background')
        
    async def create_trajectory_plot(self, line1: str, line2: str, 
                                   prediction_data: Dict) -> str:
        """
        Create enhanced 3D trajectory plot with larger prediction point and info bubble
        
        Args:
            line1: TLE line 1
            line2: TLE line 2
            prediction_data: Prediction results
            
        Returns:
            Base64 encoded plot image
        """
        try:
            # Generate reference trajectory using SGP4
            true_trajectory = await self._generate_reference_trajectory(line1, line2)
            
            # Create figure with high DPI for better quality
            fig = plt.figure(figsize=(16, 12), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot Earth
            self._plot_earth(ax)
            
            # Plot reference trajectory
            if true_trajectory:
                positions = np.array([[p['x'], p['y'], p['z']] for p in true_trajectory])
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       'cyan', alpha=0.7, linewidth=2.5, label='Current Trajectory')
            
            # Plot prediction point with MUCH larger size and enhanced styling
            pred_pos = prediction_data['prediction']['position']
            
            # Main prediction point - LARGE and prominent
            prediction_scatter = ax.scatter(
                pred_pos[0], pred_pos[1], pred_pos[2], 
                color='#FF4444',  # Bright red
                s=800,  # Much larger size (was 150)
                alpha=0.9, 
                label='Predicted Position',
                edgecolors='white',  # White border
                linewidth=3,  # Thick border
                marker='o'
            )
            
            # Add a glowing effect with multiple layers
            ax.scatter(pred_pos[0], pred_pos[1], pred_pos[2], 
                      color='#FF6666', s=1200, alpha=0.3, marker='o')  # Outer glow
            ax.scatter(pred_pos[0], pred_pos[1], pred_pos[2], 
                      color='#FF8888', s=1000, alpha=0.5, marker='o')  # Middle glow
            
            # Plot current position if available
            if 'sgp4_prediction' in prediction_data:
                current_pos = prediction_data['sgp4_prediction']['position']
                ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                          color='#00FF00', s=400, alpha=0.8, label='Current Position',
                          edgecolors='white', linewidth=2)
            
            # Customize plot
            self._customize_3d_plot(ax, 'Satellite Trajectory Prediction')
            
            # Add enhanced info text with confidence and details
            self._add_enhanced_prediction_info(fig, ax, prediction_data, pred_pos)
            
            # Convert to base64
            return self._figure_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating trajectory plot: {e}")
            return self._create_error_plot(str(e))
    
    def _add_enhanced_prediction_info(self, fig, ax, prediction_data: Dict, pred_pos: List[float]):
        """Add enhanced prediction information with bubble message"""
        
        # Extract confidence information
        confidence_info = prediction_data.get('confidence', {})
        confidence_score = confidence_info.get('overall_confidence', 0.7)
        confidence_level = confidence_info.get('confidence_level', 'MEDIUM')
        method = prediction_data.get('method', 'sgp4')
        
        # Get prediction time
        hours_ahead = prediction_data.get('hours_ahead', 24)
        target_time = datetime.utcnow() + timedelta(hours=hours_ahead)
        
        # Create bubble message text
        bubble_text = (
            f"🛰️ SATELLITE PREDICTION\n\n"
            f"📍 The satellite will be HERE\n"
            f"⏰ Time: {target_time.strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"🎯 Confidence: {confidence_score:.1%} ({confidence_level})\n"
            f"🔬 Method: {method.upper()}\n"
            f"📏 Altitude: {np.sqrt(sum(x**2 for x in pred_pos)) - 6371:.1f} km"
        )
        
        # Add bubble as annotation with enhanced styling
        try:
            # Convert 3D coordinates to 2D for annotation
            proj_x, proj_y, _ = ax.proj3d.proj_transform(pred_pos[0], pred_pos[1], pred_pos[2], ax.get_proj())
            
            # Add the bubble annotation
            ax.text2D(0.02, 0.98, bubble_text, 
                     transform=ax.transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(
                         boxstyle='round,pad=1',
                         facecolor='black',
                         edgecolor='#FF4444',
                         alpha=0.9,
                         linewidth=2
                     ),
                     color='white',
                     weight='bold')
            
            # Add a second info box with technical details
            technical_info = self._get_technical_info(prediction_data)
            ax.text2D(0.98, 0.02, technical_info,
                     transform=ax.transAxes,
                     fontsize=10,
                     verticalalignment='bottom',
                     horizontalalignment='right',
                     bbox=dict(
                         boxstyle='round,pad=0.8',
                         facecolor='#001122',
                         edgecolor='cyan',
                         alpha=0.8,
                         linewidth=1
                     ),
                     color='cyan')
            
        except Exception as e:
            logger.warning(f"Could not add 3D annotation, using fallback: {e}")
            
            # Fallback: Add text box in corner
            fig.text(0.02, 0.98, bubble_text, 
                    transform=fig.transFigure,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(
                        boxstyle='round,pad=1',
                        facecolor='black',
                        edgecolor='#FF4444',
                        alpha=0.9,
                        linewidth=2
                    ),
                    color='white',
                    weight='bold')
    
    def _get_technical_info(self, prediction_data: Dict) -> str:
        """Get technical information for the second info box"""
        
        metrics = prediction_data.get('metrics', {})
        pos_metrics = metrics.get('position_metrics', {})
        
        info_lines = []
        
        # Add method info
        method = prediction_data.get('method', 'unknown')
        info_lines.append(f"Method: {method.upper()}")
        
        # Add accuracy metrics if available
        if pos_metrics.get('rmse'):
            info_lines.append(f"Position RMSE: {pos_metrics['rmse']:.2f} km")
        
        if pos_metrics.get('mae'):
            info_lines.append(f"Position MAE: {pos_metrics['mae']:.2f} km")
        
        # Add TensorFlow status
        tf_status = prediction_data.get('tensorflow_available', False)
        ml_status = "✅ ML Enhanced" if tf_status else "⚠️ Physics Only"
        info_lines.append(f"AI Status: {ml_status}")
        
        # Add timestamp
        timestamp = datetime.utcnow().strftime('%H:%M:%S UTC')
        info_lines.append(f"Generated: {timestamp}")
        
        return '\n'.join(info_lines)
    
    def _customize_3d_plot(self, ax, title: str):
        """Customize 3D plot appearance with enhanced styling"""
        ax.set_xlabel('X (km)', fontsize=12, color='white')
        ax.set_ylabel('Y (km)', fontsize=12, color='white')
        ax.set_zlabel('Z (km)', fontsize=12, color='white')
        ax.set_title(title, fontsize=16, pad=20, color='white', weight='bold')
        
        # Set equal aspect ratio
        max_range = 15000
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Enhance grid and styling
        ax.grid(True, alpha=0.3, color='gray')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make grid lines more subtle
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Enhanced legend
        legend = ax.legend(loc='upper left', fontsize=11, 
                          fancybox=True, shadow=True,
                          facecolor='black', edgecolor='white')
        legend.get_frame().set_alpha(0.8)
        
        # Set view angle for best visibility
        ax.view_init(elev=20, azim=45)
    
    def _plot_earth(self, ax):
        """Plot enhanced Earth sphere on 3D axis"""
        # Create sphere with more detail
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        earth_radius = 6371  # km
        
        x = earth_radius * np.outer(np.cos(u), np.sin(v))
        y = earth_radius * np.outer(np.sin(u), np.sin(v))
        z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot Earth with enhanced colors
        ax.plot_surface(x, y, z, alpha=0.4, color='lightblue', 
                       linewidth=0, antialiased=True, shade=True)
        
        # Add Earth's equator line
        equator_u = np.linspace(0, 2 * np.pi, 100)
        equator_x = earth_radius * np.cos(equator_u)
        equator_y = earth_radius * np.sin(equator_u)
        equator_z = np.zeros_like(equator_x)
        ax.plot(equator_x, equator_y, equator_z, 'yellow', linewidth=2, alpha=0.7, label='Equator')
    
    async def _generate_reference_trajectory(self, line1: str, line2: str, 
                                           hours: int = 24) -> List[Dict]:
        """Generate reference trajectory using SGP4"""
        try:
            if not Satrec or not jday:
                logger.warning("SGP4 not available for trajectory generation")
                return []
                
            satellite = Satrec.twoline2rv(line1, line2)
            trajectory = []
            
            # Generate more points for smoother trajectory
            total_points = min(hours * 6, 144)  # Up to 6 points per hour, max 144
            
            for i in range(total_points):
                current_time = datetime.utcnow() + timedelta(hours=i * hours / total_points)
                jd, fr = jday(current_time.year, current_time.month, current_time.day,
                             current_time.hour, current_time.minute, current_time.second)
                
                error, position, velocity = satellite.sgp4(jd, fr)
                
                if error == 0:
                    trajectory.append({
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                        'timestamp': current_time.isoformat()
                    })
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error generating reference trajectory: {e}")
            return []
    
    def _figure_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string with high quality"""
        try:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                       dpi=120, facecolor='black', edgecolor='white',
                       transparent=False, pad_inches=0.2)
            img_buffer.seek(0)
            
            plot_data = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            plt.close(fig)
            return ""
    
    def _create_error_plot(self, error_message: str) -> str:
        """Create enhanced error plot when visualization fails"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
            ax.set_facecolor('black')
            
            # Create a more informative error display
            error_text = f"🔧 Visualization Error\n\n{error_message}\n\n💡 The prediction data is still available!\n📊 Check the metrics panel for results."
            
            ax.text(0.5, 0.5, error_text, 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, color='#FF6666',
                   bbox=dict(boxstyle='round,pad=1', facecolor='black', 
                            edgecolor='#FF4444', alpha=0.9, linewidth=2),
                   weight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            return self._figure_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating error plot: {e}")
            return ""
    
    # ... (other methods remain the same as in the original file)
    
    async def create_ground_track_plot(self, norad_id: str, 
                                     duration_hours: int = 24) -> str:
        """Create enhanced ground track visualization"""
        try:
            if not self.satellite_service:
                return self._create_error_plot("Satellite service not available")
                
            # Get ground track data
            ground_track_data = await self.satellite_service.get_ground_track(
                norad_id, duration_hours
            )
            
            # Create figure with enhanced styling
            fig, ax = plt.subplots(figsize=(18, 12), facecolor='black')
            ax.set_facecolor('black')
            
            # Plot world map outline
            self._plot_world_map(ax)
            
            # Plot ground track with enhanced styling
            ground_track = ground_track_data['ground_track']
            if ground_track:
                lats = [point['latitude'] for point in ground_track]
                lons = [point['longitude'] for point in ground_track]
                
                # Handle longitude wraparound
                lons_wrapped = self._handle_longitude_wraparound(lons)
                
                for lon_segment in lons_wrapped:
                    if len(lon_segment) > 1:
                        ax.plot(lon_segment, lats[:len(lon_segment)], 
                               'cyan', linewidth=3, alpha=0.8)
                
                # Mark start and end points with larger markers
                ax.scatter(lons[0], lats[0], color='#00FF00', s=200, 
                          label='Start', zorder=5, edgecolors='white', linewidth=2)
                ax.scatter(lons[-1], lats[-1], color='#FF4444', s=200, 
                          label='End', zorder=5, edgecolors='white', linewidth=2)
            
            # Customize plot with enhanced styling
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel('Longitude (°)', fontsize=14, color='white')
            ax.set_ylabel('Latitude (°)', fontsize=14, color='white')
            ax.set_title(f"Ground Track - {ground_track_data['name']}\n"
                        f"Duration: {duration_hours} hours", 
                        fontsize=16, pad=20, color='white', weight='bold')
            ax.grid(True, alpha=0.3, color='gray')
            ax.legend(loc='upper right', fontsize=12, facecolor='black', 
                     edgecolor='white', labelcolor='white')
            
            # Set colors for axes
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            plt.tight_layout()
            return self._figure_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating ground track plot: {e}")
            return self._create_error_plot(str(e))
    
    def _plot_world_map(self, ax):
        """Plot simple world map outline with enhanced styling"""
        # Draw enhanced grid
        for lat in range(-90, 91, 30):
            ax.axhline(lat, color='gray', alpha=0.4, linewidth=0.8)
        for lon in range(-180, 181, 30):
            ax.axvline(lon, color='gray', alpha=0.4, linewidth=0.8)
        
        # Add continent outlines (simplified)
        # This is a basic implementation - you could add more detailed coastlines
        ax.axhline(0, color='yellow', alpha=0.6, linewidth=1, label='Equator')
    
    def _handle_longitude_wraparound(self, longitudes: List[float]) -> List[List[float]]:
        """Handle longitude wraparound at ±180°"""
        segments = []
        current_segment = []
        
        for i, lon in enumerate(longitudes):
            if i > 0:
                prev_lon = longitudes[i-1]
                # Check for wraparound
                if abs(lon - prev_lon) > 180:
                    # End current segment and start new one
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = [lon]
                else:
                    current_segment.append(lon)
            else:
                current_segment.append(lon)
        
        if current_segment:
            segments.append(current_segment)
        
        return segments