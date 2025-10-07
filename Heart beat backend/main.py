import cv2
import numpy as np
from scipy import signal, fft
from scipy.signal import find_peaks, cheby2
import asyncio
import base64
from fastapi import FastAPI
import socketio
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# --- Strict Configuration for Real vs Fake Detection ---
BUFFER_SECONDS = 12  # Longer buffer for better analysis
FPS = 10
BUFFER_SIZE = BUFFER_SECONDS * FPS
BPM_RANGE = [50, 140]

# STRICT thresholds for real human detection
SIGNAL_QUALITY_THRESHOLD = 0.4  # High SNR requirement
TMC_THRESHOLD = 0.8  # High template matching requirement
MIN_FACE_SIZE = 120
MIN_CONFIDENCE_THRESHOLD = 0.85  # Very high confidence requirement
MIN_VALID_REGIONS = 3  # Require multiple regions
PULSE_CONSISTENCY_THRESHOLD = 0.7  # Pulse must be consistent across regions
TEMPORAL_CONSISTENCY_THRESHOLD = 0.8  # Temporal consistency check
MICRO_MOVEMENT_THRESHOLD = 2.0  # Minimum micro-movements required

# Multi-region configuration - more regions for better detection
REGIONS_CONFIG = {
    'forehead': {'x_start': 0.25, 'x_end': 0.75, 'y_start': 0.15, 'y_end': 0.35},
    'left_cheek': {'x_start': 0.1, 'x_end': 0.45, 'y_start': 0.4, 'y_end': 0.75},
    'right_cheek': {'x_start': 0.55, 'x_end': 0.9, 'y_start': 0.4, 'y_end': 0.75},
    'nose_bridge': {'x_start': 0.4, 'x_end': 0.6, 'y_start': 0.3, 'y_end': 0.5},
    'chin': {'x_start': 0.3, 'x_end': 0.7, 'y_start': 0.75, 'y_end': 0.95},
    'left_temple': {'x_start': 0.05, 'x_end': 0.3, 'y_start': 0.2, 'y_end': 0.4},
    'right_temple': {'x_start': 0.7, 'x_end': 0.95, 'y_start': 0.2, 'y_end': 0.4}
}

@dataclass
class EnhancedSignalQuality:
    snr: float
    tmc: float
    peak_frequency: float
    confidence: float
    pulse_consistency: float
    temporal_stability: float
    micro_movements: float
    region_correlation: float

@dataclass
class LivenessMetrics:
    has_pulse_variation: bool
    has_micro_movements: bool
    has_natural_noise: bool
    temporal_consistency: float
    spatial_consistency: float
    overall_liveness_score: float

# --- Load Face Detection Model ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- Enhanced Global State per Connection ---
session_data = {}

# --- FastAPI and Socket.IO Setup ---
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*', max_http_buffer_size=10000000)
asgi_app = socketio.ASGIApp(sio, app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrictLivenessDetector:
    """Strict liveness detection to differentiate real humans from photos/AI"""
    
    @staticmethod
    def detect_micro_movements(current_frame: np.ndarray, previous_frames: List[np.ndarray], face_coords: Tuple[int, int, int, int]) -> float:
        """Detect subtle micro-movements that indicate a live person"""
        if len(previous_frames) < 5:
            return 0.0
        
        try:
            x, y, w, h = face_coords
            current_roi = current_frame[y:y+h, x:x+w]
            
            movements = []
            for prev_frame in previous_frames[-5:]:
                prev_roi = prev_frame[y:y+h, x:x+w]
                
                # Calculate optical flow
                current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
                
                # Detect corners for tracking
                corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
                
                if corners is not None and len(corners) > 10:
                    # Calculate optical flow
                    next_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, corners, None)
                    
                    # Calculate movement magnitude
                    good_corners = corners[status == 1]
                    good_next = next_corners[status == 1]
                    
                    if len(good_corners) > 5:
                        movement_vectors = good_next - good_corners
                        movement_magnitude = np.mean(np.linalg.norm(movement_vectors, axis=1))
                        movements.append(movement_magnitude)
            
            return np.mean(movements) if movements else 0.0
            
        except Exception as e:
            logger.error(f"Micro-movement detection error: {e}")
            return 0.0
    
    @staticmethod
    def analyze_pulse_consistency_across_regions(region_signals: Dict[str, List[float]]) -> float:
        """Analyze if pulse signals are consistent across different facial regions"""
        try:
            if len(region_signals) < 3:
                return 0.0
            
            # Get signals that have enough data
            valid_signals = {k: v for k, v in region_signals.items() if len(v) >= BUFFER_SIZE}
            
            if len(valid_signals) < 3:
                return 0.0
            
            correlations = []
            signal_names = list(valid_signals.keys())
            
            # Calculate cross-correlations between all region pairs
            for i in range(len(signal_names)):
                for j in range(i + 1, len(signal_names)):
                    sig1 = np.array(valid_signals[signal_names[i]])
                    sig2 = np.array(valid_signals[signal_names[j]])
                    
                    # Normalize signals
                    sig1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-8)
                    sig2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-8)
                    
                    # Calculate correlation
                    correlation = np.corrcoef(sig1, sig2)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Pulse consistency analysis error: {e}")
            return 0.0
    
    @staticmethod
    def detect_natural_physiological_noise(signal_data: np.ndarray, fs: float) -> bool:
        """Detect natural physiological noise patterns that are absent in fake signals"""
        try:
            # Real physiological signals have natural variations and noise
            # 1. Check for natural frequency variations
            signal_diff = np.diff(signal_data)
            variation_coefficient = np.std(signal_diff) / (np.mean(np.abs(signal_diff)) + 1e-8)
            
            # 2. Check for natural harmonics in frequency domain
            fft_data = np.abs(fft.fft(signal_data))
            freqs = fft.fftfreq(len(signal_data), 1.0/fs)
            
            # Real pulse signals should have harmonics at 2x, 3x the fundamental frequency
            fundamental_idx = np.argmax(fft_data[:len(fft_data)//2])
            fundamental_freq = freqs[fundamental_idx]
            
            if fundamental_freq > 0:
                # Look for harmonics
                harmonic_2_idx = int(fundamental_idx * 2)
                harmonic_3_idx = int(fundamental_idx * 3)
                
                if harmonic_2_idx < len(fft_data) and harmonic_3_idx < len(fft_data):
                    harmonic_strength = (fft_data[harmonic_2_idx] + fft_data[harmonic_3_idx]) / fft_data[fundamental_idx]
                    has_harmonics = harmonic_strength > 0.1
                else:
                    has_harmonics = False
            else:
                has_harmonics = False
            
            # 3. Check for natural noise floor
            noise_floor = np.percentile(fft_data, 25)  # Bottom 25% as noise
            signal_peak = np.max(fft_data)
            dynamic_range = signal_peak / (noise_floor + 1e-8)
            
            # Real signals should have: variation, harmonics, and reasonable dynamic range
            return (variation_coefficient > 0.1 and 
                   has_harmonics and 
                   5 < dynamic_range < 500)
            
        except Exception as e:
            logger.error(f"Natural noise detection error: {e}")
            return False
    
    @staticmethod
    def calculate_temporal_stability(signal_buffer: List[float], window_size: int = 30) -> float:
        """Calculate temporal stability - real humans have consistent but varying pulse"""
        try:
            if len(signal_buffer) < window_size * 2:
                return 0.0
            
            # Divide signal into windows and analyze stability
            windows = []
            for i in range(0, len(signal_buffer) - window_size, window_size // 2):
                window = signal_buffer[i:i + window_size]
                windows.append(np.std(window))
            
            if len(windows) < 2:
                return 0.0
            
            # Real signals should have consistent variability
            stability = 1.0 - (np.std(windows) / (np.mean(windows) + 1e-8))
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.error(f"Temporal stability calculation error: {e}")
            return 0.0

class EnhancedRPPGProcessor:
    """Enhanced rPPG processor with strict liveness detection"""
    
    @staticmethod
    def extract_multi_region_signals(frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract rPPG signals from multiple facial regions with better preprocessing"""
        x, y, w, h = face_coords
        signals = {}
        
        for region_name, coords in REGIONS_CONFIG.items():
            roi_x_start = x + int(w * coords['x_start'])
            roi_x_end = x + int(w * coords['x_end'])
            roi_y_start = y + int(h * coords['y_start'])
            roi_y_end = y + int(h * coords['y_end'])
            
            # Ensure ROI is within frame bounds
            roi_x_start = max(0, roi_x_start)
            roi_x_end = min(frame.shape[1], roi_x_end)
            roi_y_start = max(0, roi_y_start)
            roi_y_end = min(frame.shape[0], roi_y_end)
            
            roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            
            if roi.size > 100:  # Ensure ROI is large enough
                # Use Green-Red channel difference (more robust for pulse detection)
                green_channel = np.mean(roi[:, :, 1])
                red_channel = np.mean(roi[:, :, 0])
                
                # ChromaGan method: (Green - Red) / (Green + Red)
                if (green_channel + red_channel) > 0:
                    signal_value = (green_channel - red_channel) / (green_channel + red_channel)
                else:
                    signal_value = 0.0
                
                signals[region_name] = signal_value
            else:
                signals[region_name] = 0.0
                
        return signals
    
    @staticmethod
    def calculate_template_match_correlation(signal_data: np.ndarray, fs: float) -> float:
        """Calculate Template Match Correlation with stricter requirements - FIXED LOCATION"""
        try:
            min_distance = int(fs * 0.6)  # More restrictive peak detection
            peaks, properties = find_peaks(signal_data, distance=min_distance, prominence=0.1)
            
            if len(peaks) < 4:  # Need more peaks for reliable analysis
                return 0.0
            
            intervals = np.diff(peaks)
            median_interval = np.median(intervals)
            
            # Check interval consistency (real hearts have consistent rhythm)
            interval_consistency = 1.0 - (np.std(intervals) / (median_interval + 1e-8))
            if interval_consistency < 0.7:
                return 0.0
            
            window_size = int(median_interval)
            pulses = []
            
            for peak in peaks:
                start = max(0, peak - window_size//2)
                end = min(len(signal_data), peak + window_size//2)
                if end - start == window_size:
                    pulse = signal_data[start:end]
                    # Normalize pulse
                    pulse = (pulse - np.mean(pulse)) / (np.std(pulse) + 1e-8)
                    pulses.append(pulse)
            
            if len(pulses) < 3:
                return 0.0
            
            template = np.mean(pulses, axis=0)
            correlations = []
            
            for pulse in pulses:
                corr = np.corrcoef(pulse, template)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"TMC calculation error: {e}")
            return 0.0
    
    @staticmethod
    def enhanced_signal_quality_assessment(signal_data: np.ndarray, fs: float, region_signals: Dict[str, List[float]], previous_frames: List[np.ndarray], current_frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Optional[EnhancedSignalQuality]:
        """Comprehensive signal quality assessment with liveness detection"""
        try:
            # Standard rPPG processing
            detrended = signal.detrend(signal_data)
            
            # Improved filtering
            low_freq = BPM_RANGE[0] / 60.0
            high_freq = BPM_RANGE[1] / 60.0
            nyquist = fs / 2
            
            if low_freq >= nyquist or high_freq >= nyquist:
                return None
            
            b, a = cheby2(6, 40, [low_freq/nyquist, high_freq/nyquist], btype='band')
            filtered_signal = signal.filtfilt(b, a, detrended)
            
            # FFT analysis
            fft_data = np.abs(fft.fft(filtered_signal))
            freqs = fft.fftfreq(len(filtered_signal), 1.0/fs)
            
            valid_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
            if len(valid_indices) == 0:
                return None
            
            peak_index = np.argmax(fft_data[valid_indices])
            peak_freq = freqs[valid_indices][peak_index]
            peak_power = fft_data[valid_indices][peak_index]
            
            # Enhanced SNR calculation
            total_power = np.sum(fft_data[valid_indices])
            noise_power = total_power - peak_power
            snr = peak_power / noise_power if noise_power > 0 else 0.0
            
            # FIXED: Call from correct class
            tmc = EnhancedRPPGProcessor.calculate_template_match_correlation(filtered_signal, fs)
            
            # NEW: Liveness-specific assessments
            pulse_consistency = StrictLivenessDetector.analyze_pulse_consistency_across_regions(region_signals)
            temporal_stability = StrictLivenessDetector.calculate_temporal_stability(signal_data.tolist())
            micro_movements = StrictLivenessDetector.detect_micro_movements(current_frame, previous_frames, face_coords)
            
            # Region correlation (how well different regions correlate)
            region_correlation = pulse_consistency
            
            # Natural physiological patterns
            has_natural_patterns = StrictLivenessDetector.detect_natural_physiological_noise(filtered_signal, fs)
            
            # STRICT confidence calculation
            bpm = peak_freq * 60
            
            # BPM must be in reasonable range
            if not (BPM_RANGE[0] <= bpm <= BPM_RANGE[1]):
                bpm_confidence = 0.0
            else:
                bpm_confidence = 1.0
            
            # All metrics must pass strict thresholds
            snr_confidence = 1.0 if snr >= SIGNAL_QUALITY_THRESHOLD else 0.0
            tmc_confidence = 1.0 if tmc >= TMC_THRESHOLD else 0.0
            consistency_confidence = 1.0 if pulse_consistency >= PULSE_CONSISTENCY_THRESHOLD else 0.0
            temporal_confidence = 1.0 if temporal_stability >= TEMPORAL_CONSISTENCY_THRESHOLD else 0.0
            movement_confidence = 1.0 if micro_movements >= MICRO_MOVEMENT_THRESHOLD else 0.0
            natural_confidence = 1.0 if has_natural_patterns else 0.0
            
            # ALL criteria must be met for high confidence
            overall_confidence = (bmp_confidence * snr_confidence * tmc_confidence * 
                                consistency_confidence * temporal_confidence * 
                                movement_confidence * natural_confidence)
            
            return EnhancedSignalQuality(
                snr=snr,
                tmc=tmc,
                peak_frequency=peak_freq,
                confidence=overall_confidence,
                pulse_consistency=pulse_consistency,
                temporal_stability=temporal_stability,
                micro_movements=micro_movements,
                region_correlation=region_correlation
            )
            
        except Exception as e:
            logger.error(f"Enhanced signal quality assessment error: {e}")
            return None

# --- Strict Processing Logic ---
async def strict_process_heartbeat(sid: str) -> Optional[Dict]:
    """Strict heartbeat processing that can differentiate real humans from fakes"""
    data = session_data.get(sid)
    if not data or len(data['combined_signals']) < BUFFER_SIZE:
        return None
    
    logger.info(f"[{sid}] Processing with STRICT liveness detection...")
    
    try:
        region_results = {}
        valid_regions = 0
        
        # Get frames for micro-movement analysis
        previous_frames = data.get('frame_buffer', [])[-10:]  # Last 10 frames
        current_frame = data.get('last_frame')
        face_coords = data.get('last_face_coords')
        
        if current_frame is None or not face_coords:
            return None
        
        # Process each region with strict criteria
        for region_name in REGIONS_CONFIG.keys():
            if region_name in data and len(data[region_name]) >= BUFFER_SIZE:
                region_signal = np.array(data[region_name])
                timestamps = np.array(data['timestamps'])
                
                fs = 1 / np.mean(np.diff(timestamps)) if len(timestamps) > 1 else FPS
                
                # Enhanced assessment with liveness detection
                quality = EnhancedRPPGProcessor.enhanced_signal_quality_assessment(
                    region_signal, fs, data, previous_frames, current_frame, face_coords
                )
                
                if quality and quality.confidence > 0.9:  # Very strict threshold
                    bpm = quality.peak_frequency * 60
                    region_results[region_name] = {
                        'bpm': bpm,
                        'quality': quality
                    }
                    valid_regions += 1
        
        # STRICT VALIDATION: Require multiple valid regions
        if valid_regions < MIN_VALID_REGIONS:
            logger.info(f"[{sid}] Failed strict validation - only {valid_regions} valid regions")
            return {
                'status': 'suspicious',
                'reason': 'Insufficient biological signal consistency',
                'valid_regions': valid_regions,
                'confidence': 0.0
            }
        
        # Calculate consensus BPM
        bpms = [result['bpm'] for result in region_results.values()]
        bpm_std = np.std(bpms)
        bpm_mean = np.mean(bpms)
        
        # BPM must be consistent across regions (real humans have consistent pulse)
        if bpm_std > 5.0:  # Too much variation between regions
            return {
                'status': 'suspicious',
                'reason': 'Inconsistent pulse across facial regions',
                'bmp_variation': bmp_std,
                'confidence': 0.0
            }
        
        # Calculate overall metrics
        avg_quality = np.mean([result['quality'].confidence for result in region_results.values()])
        avg_snr = np.mean([result['quality'].snr for result in region_results.values()])
        avg_tmc = np.mean([result['quality'].tmc for result in region_results.values()])
        avg_consistency = np.mean([result['quality'].pulse_consistency for result in region_results.values()])
        avg_temporal = np.mean([result['quality'].temporal_stability for result in region_results.values()])
        avg_movements = np.mean([result['quality'].micro_movements for result in region_results.values()])
        
        # FINAL STRICT VALIDATION
        is_authentic = (
            BPM_RANGE[0] <= bmp_mean <= BPM_RANGE[1] and
            avg_quality >= 0.9 and  # Very high confidence required
            avg_snr >= SIGNAL_QUALITY_THRESHOLD and
            avg_tmc >= TMC_THRESHOLD and
            avg_consistency >= PULSE_CONSISTENCY_THRESHOLD and
            avg_temporal >= TEMPORAL_CONSISTENCY_THRESHOLD and
            avg_movements >= MICRO_MOVEMENT_THRESHOLD and
            valid_regions >= MIN_VALID_REGIONS and
            bmp_std <= 5.0
        )
        
        if is_authentic:
            return {
                'status': 'authentic',
                'bpm': round(bpm_mean, 2),
                'confidence': round(avg_quality, 3),
                'snr': round(avg_snr, 3),
                'tmc': round(avg_tmc, 3),
                'consistency': round(avg_consistency, 3),
                'temporal_stability': round(avg_temporal, 3),
                'micro_movements': round(avg_movements, 3),
                'valid_regions': valid_regions,
                'bmp_consistency': round(bmp_std, 2)
            }
        else:
            return {
                'status': 'suspicious',
                'reason': 'Failed strict liveness validation',
                'bpm': round(bpm_mean, 2) if 'bpm_mean' in locals() else 0,
                'confidence': round(avg_quality, 3) if 'avg_quality' in locals() else 0,
                'snr': round(avg_snr, 3) if 'avg_snr' in locals() else 0,
                'valid_regions': valid_regions,
                'details': {
                    'quality': avg_quality >= 0.9,
                    'snr': avg_snr >= SIGNAL_QUALITY_THRESHOLD,
                    'tmc': avg_tmc >= TMC_THRESHOLD,
                    'consistency': avg_consistency >= PULSE_CONSISTENCY_THRESHOLD,
                    'temporal': avg_temporal >= TEMPORAL_CONSISTENCY_THRESHOLD,
                    'movements': avg_movements >= MICRO_MOVEMENT_THRESHOLD
                }
            }
            
    except Exception as e:
        logger.error(f"[{sid}] Strict processing error: {e}")
        return None

# --- Socket.IO Event Handlers ---
@sio.on('connect')
async def connect(sid, environ):
    logger.info(f'Client connected: {sid}')
    session_data[sid] = {
        'combined_signals': [],
        'timestamps': [],
        'last_frame': None,
        'last_face_coords': None,
        'frame_buffer': []  # Store recent frames for micro-movement analysis
    }
    
    # Initialize region-specific buffers
    for region_name in REGIONS_CONFIG.keys():
        session_data[sid][region_name] = []

@sio.on('video_frame')
async def strict_video_frame(sid, data):
    """Strict video frame processing with comprehensive liveness detection"""
    try:
        # Decode frame
        header, encoded = data.split(",", 1)
        frame_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # Store frame for micro-movement analysis
        session_data[sid]['last_frame'] = frame.copy()
        session_data[sid]['frame_buffer'].append(frame.copy())
        
        # Keep only recent frames
        if len(session_data[sid]['frame_buffer']) > 15:
            session_data[sid]['frame_buffer'].pop(0)
        
        # Face detection with stricter parameters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 6, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        
        if len(faces) > 0:
            # Use the largest detected face
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            session_data[sid]['last_face_coords'] = (x, y, w, h)
            
            # Extract multi-region signals
            region_signals = EnhancedRPPGProcessor.extract_multi_region_signals(frame, (x, y, w, h))
            
            # Store signals
            current_time = asyncio.get_event_loop().time()
            session_data[sid]['timestamps'].append(current_time)
            
            # Calculate combined signal
            valid_signals = [v for v in region_signals.values() if abs(v) > 1e-6]
            if valid_signals:
                combined_signal = np.mean(valid_signals)
                session_data[sid]['combined_signals'].append(combined_signal)
            
            # Store individual region signals
            for region_name, signal_value in region_signals.items():
                if len(session_data[sid][region_name]) >= BUFFER_SIZE:
                    session_data[sid][region_name].pop(0)
                session_data[sid][region_name].append(signal_value)
            
            # Update progress
            progress = len(session_data[sid]['combined_signals']) / BUFFER_SIZE * 100
            await sio.emit('status_update', {
                'message': f'Analyzing biological patterns... {int(progress)}%',
                'regions_detected': len(valid_signals),
                'buffer_length': len(session_data[sid]['combined_signals'])
            }, to=sid)
            
            # Process when buffer is full
            if len(session_data[sid]['combined_signals']) >= BUFFER_SIZE:
                result = await strict_process_heartbeat(sid)
                
                if result:
                    if result['status'] == 'authentic':
                        await sio.emit('verification_result', {
                            'status': 'Verified - Authentic Human',
                            'bmp': result['bpm'],
                            'confidence': result['confidence'],
                            'snr': result['snr'],
                            'tmc': result['tmc'],
                            'valid_regions': result['valid_regions'],
                            'analysis_type': 'strict-liveness',
                            'consistency_score': result['consistency'],
                            'temporal_stability': result['temporal_stability'],
                            'micro_movements': result['micro_movements']
                        }, to=sid)
                    else:
                        await sio.emit('verification_result', {
                            'status': 'Suspicious - Possible Fake',
                            'reason': result['reason'],
                            'confidence': result['confidence'],
                            'valid_regions': result['valid_regions'],
                            'analysis_type': 'strict-liveness',
                            'details': result.get('details', {})
                        }, to=sid)
                else:
                    await sio.emit('verification_result', {
                        'status': 'Failed',
                        'reason': 'Unable to detect authentic biological patterns'
                    }, to=sid)
                
                # Reset buffers
                session_data[sid]['combined_signals'] = []
                session_data[sid]['timestamps'] = []
                for region_name in REGIONS_CONFIG.keys():
                    session_data[sid][region_name] = []
        else:
            await sio.emit('status_update', {
                'message': 'No face detected. Please position yourself clearly in the frame.'
            }, to=sid)
            
            # Reset buffers when no face detected
            session_data[sid]['combined_signals'] = []
            session_data[sid]['timestamps'] = []
            for region_name in REGIONS_CONFIG.keys():
                session_data[sid][region_name] = []
                
    except Exception as e:
        logger.error(f"Strict frame processing error for {sid}: {e}")
        await sio.emit('error', {'message': 'Frame processing error'}, to=sid)

@sio.on('disconnect')
async def disconnect(sid):
    logger.info(f'Client disconnected: {sid}')
    if sid in session_data:
        del session_data[sid]

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "3.0.0", "features": ["strict-liveness", "anti-spoof", "multi-region"]}

# --- Main Application ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)
