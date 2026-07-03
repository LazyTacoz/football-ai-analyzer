/**
 * Football AI Analyzer - Main Application (v3.0)
 * ===============================================
 * New in v3.0:
 * - Live speed ticker (km/h per player)
 * - Team distance panel (total metres per team)
 * - Sprint alert flash notifications
 * - Auto homography status badge
 * - Top speeds leaderboard
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Play, Pause, Upload, Activity, Users, Target, Eye, EyeOff,
  Settings, Radar, Hexagon, Circle, Loader2, Check, AlertCircle,
  ChevronRight, Footprints, Zap, BarChart3, Download, User, Hash,
  Flame, X, ChevronDown, MapPin, Move, Navigation, Timer, TrendingUp,
  Gauge, Route
} from 'lucide-react';

// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE_URL = 'http://localhost:8000';
const WS_BASE_URL  = 'ws://localhost:8000';

// ============================================================
// CUSTOM HOOKS
// ============================================================

function useWebSocket(videoId, onMessage) {
  const wsRef        = useRef(null);
  const reconnectRef = useRef(null);

  useEffect(() => {
    if (!videoId) return;

    const connect = () => {
      const ws = new WebSocket(`${WS_BASE_URL}/ws/${videoId}`);
      ws.onopen    = () => console.log('WebSocket connected');
      ws.onmessage = (e) => onMessage(JSON.parse(e.data));
      ws.onclose   = () => { reconnectRef.current = setTimeout(connect, 2000); };
      ws.onerror   = (err) => console.error('WS error:', err);
      wsRef.current = ws;
    };

    connect();

    const ping = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send('ping');
    }, 30000);

    return () => {
      clearInterval(ping);
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [videoId, onMessage]);

  return wsRef;
}

// ============================================================
// SPEED HELPERS
// ============================================================

function speedColor(kmh) {
  if (kmh >= 25) return '#ef4444'; // sprint  → red
  if (kmh >= 14) return '#f97316'; // run     → orange
  if (kmh >= 7)  return '#22c55e'; // jog     → green
  return '#6b7280';                 // walk    → grey
}

function speedLabel(kmh) {
  if (kmh >= 25) return 'Sprint';
  if (kmh >= 14) return 'Run';
  if (kmh >= 7)  return 'Jog';
  return 'Walk';
}

// ============================================================
// UI PRIMITIVES
// ============================================================

function ToggleSwitch({ enabled, onChange, label, icon: Icon }) {
  return (
    <label className="flex items-center justify-between cursor-pointer group">
      <div className="flex items-center gap-3">
        {Icon && <Icon className="w-4 h-4 text-surface-400 group-hover:text-pitch-400 transition-colors" />}
        <span className="text-sm font-medium text-surface-300 group-hover:text-surface-100 transition-colors">
          {label}
        </span>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        onClick={() => onChange(!enabled)}
        className={`relative w-11 h-6 rounded-full transition-colors ${enabled ? 'bg-pitch-500' : 'bg-surface-600'}`}
      >
        <span className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform ${enabled ? 'translate-x-5' : 'translate-x-0'}`} />
      </button>
    </label>
  );
}

function StatCard({ label, value, icon: Icon, color = 'pitch', subValue }) {
  const colors = {
    pitch:  'text-pitch-400',
    red:    'text-red-400',
    blue:   'text-blue-400',
    yellow: 'text-yellow-400',
    orange: 'text-orange-400',
  };

  return (
    <div className="p-4 rounded-xl bg-surface-800/50 border border-surface-700/50 hover:border-surface-600 transition-colors">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-surface-400 uppercase tracking-wider mb-1">{label}</p>
          <p className={`text-2xl font-bold font-display ${colors[color]}`}>{value}</p>
          {subValue && <p className="text-xs text-surface-500 mt-1">{subValue}</p>}
        </div>
        <div className={`p-2 rounded-lg bg-surface-800/50 ${colors[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
}

function PossessionBar({ teamA, teamB }) {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center text-sm">
        <div className="flex items-center gap-2 text-red-400">
          <Circle className="w-3 h-3 fill-current" />
          <span>Team A</span>
          <span className="font-bold">{teamA.toFixed(1)}%</span>
        </div>
        <div className="flex items-center gap-2 text-blue-400">
          <span className="font-bold">{teamB.toFixed(1)}%</span>
          <span>Team B</span>
          <Circle className="w-3 h-3 fill-current" />
        </div>
      </div>
      <div className="h-3 rounded-full overflow-hidden flex bg-surface-700">
        <div className="bg-gradient-to-r from-red-600 to-red-500 transition-all duration-500" style={{ width: `${teamA}%` }} />
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500" style={{ width: `${teamB}%` }} />
      </div>
    </div>
  );
}

function ProgressBar({ progress, status }) {
  const colors = {
    pending:    'bg-surface-600',
    uploading:  'bg-yellow-500',
    processing: 'bg-pitch-500',
    completed:  'bg-pitch-400',
    error:      'bg-red-500',
  };
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-surface-400 capitalize">{status}</span>
        <span className="text-surface-300 font-mono">{progress.toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-surface-700 overflow-hidden">
        <div className={`h-full transition-all duration-300 ${colors[status]}`} style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}

function VideoDropzone({ onUpload, isUploading }) {
  const onDrop = useCallback((files) => { if (files.length > 0) onUpload(files[0]); }, [onUpload]);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'] },
    maxFiles: 1,
    disabled: isUploading,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
        isDragActive
          ? 'border-pitch-400 bg-pitch-500/10'
          : 'border-surface-600 hover:border-surface-500 bg-surface-800/30'
      }`}
    >
      <input {...getInputProps()} />
      <div className="space-y-4">
        {isUploading ? (
          <Loader2 className="w-16 h-16 mx-auto text-pitch-400 animate-spin" />
        ) : (
          <div className="relative mx-auto w-16 h-16">
            <div className="absolute inset-0 bg-pitch-500/20 rounded-full animate-ping" />
            <div className="relative bg-surface-800 rounded-full p-4 border border-surface-600">
              <Upload className="w-8 h-8 text-pitch-400" />
            </div>
          </div>
        )}
        <div>
          <p className="text-lg font-medium text-surface-200">
            {isDragActive ? 'Drop your video here' : 'Drag & drop a match video'}
          </p>
          <p className="text-sm text-surface-500 mt-1">or click to browse • MP4, AVI, MOV supported</p>
        </div>
      </div>
    </div>
  );
}

function VideoPlayer({ frameData, isProcessing }) {
  if (!frameData && !isProcessing) {
    return (
      <div className="aspect-video bg-surface-900 rounded-xl flex items-center justify-center border border-surface-700">
        <div className="text-center text-surface-500">
          <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p className="text-sm">No video loaded</p>
        </div>
      </div>
    );
  }
  if (isProcessing && !frameData) {
    return (
      <div className="aspect-video bg-surface-900 rounded-xl flex items-center justify-center border border-surface-700">
        <div className="text-center">
          <Loader2 className="w-12 h-12 mx-auto mb-3 text-pitch-400 animate-spin" />
          <p className="text-sm text-surface-400">Initializing processor...</p>
        </div>
      </div>
    );
  }
  return (
    <div className="aspect-video bg-black rounded-xl overflow-hidden border border-surface-700">
      <img src={`data:image/jpeg;base64,${frameData}`} alt="Processed frame" className="w-full h-full object-contain" />
    </div>
  );
}

// ============================================================
// NEW: SPRINT ALERT TOAST
// ============================================================

function SprintAlertToast({ alerts, onDismiss }) {
  // Show only the latest alert
  const latest = alerts[alerts.length - 1];
  if (!latest) return null;

  const teamColor = latest.team === 0 ? 'border-red-500/60 bg-red-500/10' : 'border-blue-500/60 bg-blue-500/10';
  const textColor = latest.team === 0 ? 'text-red-300' : 'text-blue-300';

  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${teamColor} animate-pulse`}>
      <Zap className="w-4 h-4 text-yellow-400 shrink-0" />
      <div className="flex-1 min-w-0">
        <p className={`text-sm font-bold ${textColor}`}>
          #{latest.player_id} Sprint — {latest.speed_kmh} km/h
        </p>
        <p className="text-xs text-surface-500">
          {latest.team === 0 ? 'Team A' : 'Team B'} · High intensity run detected
        </p>
      </div>
      <button onClick={onDismiss} className="shrink-0 text-surface-500 hover:text-surface-300 transition-colors">
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

// ============================================================
// NEW: SPEED TICKER
// ============================================================

function SpeedTicker({ playerSpeeds, trackedIds }) {
  // Only show players currently on pitch (have a tracker ID)
  const visible = Object.entries(playerSpeeds || {})
    .filter(([id]) => !trackedIds || trackedIds.includes(Number(id)))
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10);

  if (visible.length === 0) {
    return (
      <div className="text-center py-4 text-surface-500 text-xs">
        Waiting for player data...
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {visible.map(([id, kmh]) => {
        const pct   = Math.min((kmh / 35) * 100, 100);
        const color = speedColor(kmh);
        return (
          <div key={id} className="flex items-center gap-2">
            {/* Player ID */}
            <span className="text-xs font-mono text-surface-400 w-6 text-right shrink-0">
              #{id}
            </span>
            {/* Bar */}
            <div className="flex-1 h-2 rounded-full bg-surface-700 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${pct}%`, backgroundColor: color }}
              />
            </div>
            {/* Value */}
            <span className="text-xs font-mono w-14 text-right shrink-0" style={{ color }}>
              {kmh.toFixed(1)} <span className="text-surface-600">km/h</span>
            </span>
          </div>
        );
      })}

      {/* Intensity legend */}
      <div className="flex items-center gap-3 pt-2 border-t border-surface-700/50">
        {[
          { label: 'Sprint', color: '#ef4444', min: '25+' },
          { label: 'Run',    color: '#f97316', min: '14+' },
          { label: 'Jog',    color: '#22c55e', min: '7+'  },
        ].map(({ label, color, min }) => (
          <div key={label} className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-xs text-surface-500">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================
// NEW: TEAM DISTANCE PANEL
// ============================================================

function TeamDistancePanel({ teamStats }) {
  const a = teamStats?.[0] ?? {};
  const b = teamStats?.[1] ?? {};

  const totalA = a.total_distance_m ?? 0;
  const totalB = b.total_distance_m ?? 0;
  const maxDist = Math.max(totalA, totalB, 1);

  return (
    <div className="space-y-4">
      {/* Team A */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Circle className="w-3 h-3 fill-red-400 text-red-400" />
            <span className="text-xs font-medium text-surface-300">Team A</span>
          </div>
          <span className="text-xs font-mono text-red-400">
            {(totalA / 1000).toFixed(2)} km
          </span>
        </div>
        <div className="h-2 rounded-full bg-surface-700 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-red-600 to-red-400 transition-all duration-500"
            style={{ width: `${(totalA / maxDist) * 100}%` }}
          />
        </div>
        <div className="flex gap-4 text-xs text-surface-500">
          <span>Avg {a.avg_speed_kmh?.toFixed(1) ?? '—'} km/h</span>
          <span>{a.sprint_count ?? 0} sprints</span>
          <span>{a.player_count ?? 0} players</span>
        </div>
      </div>

      {/* Team B */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Circle className="w-3 h-3 fill-blue-400 text-blue-400" />
            <span className="text-xs font-medium text-surface-300">Team B</span>
          </div>
          <span className="text-xs font-mono text-blue-400">
            {(totalB / 1000).toFixed(2)} km
          </span>
        </div>
        <div className="h-2 rounded-full bg-surface-700 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-500"
            style={{ width: `${(totalB / maxDist) * 100}%` }}
          />
        </div>
        <div className="flex gap-4 text-xs text-surface-500">
          <span>Avg {b.avg_speed_kmh?.toFixed(1) ?? '—'} km/h</span>
          <span>{b.sprint_count ?? 0} sprints</span>
          <span>{b.player_count ?? 0} players</span>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// NEW: TOP SPEEDS LEADERBOARD
// ============================================================

function TopSpeedsLeaderboard({ topSpeeds }) {
  if (!topSpeeds || topSpeeds.length === 0) {
    return <p className="text-xs text-surface-500 text-center py-2">No data yet</p>;
  }

  const medals = ['🥇', '🥈', '🥉'];

  return (
    <div className="space-y-1.5">
      {topSpeeds.map((entry, i) => {
        const teamColor = entry.team === 0 ? 'text-red-400' : 'text-blue-400';
        return (
          <div key={entry.player_id} className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-surface-800/40 transition-colors">
            <span className="text-sm w-5 shrink-0">{medals[i] ?? `${i + 1}.`}</span>
            <span className={`text-xs font-mono font-bold ${teamColor} w-8 shrink-0`}>
              #{entry.player_id}
            </span>
            <div className="flex-1 h-1.5 rounded-full bg-surface-700 overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${(entry.peak_speed_kmh / 38) * 100}%`,
                  backgroundColor: speedColor(entry.peak_speed_kmh),
                }}
              />
            </div>
            <span className="text-xs font-mono text-surface-300 w-16 text-right shrink-0">
              {entry.peak_speed_kmh.toFixed(1)} km/h
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// PLAYER RADAR
// ============================================================

function PlayerRadar({ radarData, stats, onPlayerClick }) {
  const playerIds  = stats?.active_player_ids ?? [];
  const teamACount = stats?.team_a_count ?? 0;
  const teamBCount = stats?.team_b_count ?? 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider flex items-center gap-2">
          <Radar className="w-4 h-4 text-pitch-400" />
          Player Radar
        </h3>
        <span className="text-xs text-surface-500">{playerIds.length} tracked</span>
      </div>

      <div className="relative bg-gradient-to-b from-green-900/40 to-green-800/40 rounded-xl overflow-hidden border border-green-700/30">
        {radarData ? (
          <img src={`data:image/jpeg;base64,${radarData}`} alt="Tactical radar" className="w-full h-auto" />
        ) : (
          <div className="aspect-[7/4.5] flex items-center justify-center">
            <div className="text-center text-green-300/50">
              <MapPin className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-xs">Waiting for data...</p>
            </div>
          </div>
        )}
        <div className="absolute bottom-2 left-2 right-2 flex justify-between">
          <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-lg">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-xs text-white">Team A ({teamACount})</span>
          </div>
          <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-lg">
            <span className="text-xs text-white">Team B ({teamBCount})</span>
            <div className="w-3 h-3 rounded-full bg-blue-500" />
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-xs text-surface-400 flex items-center gap-1">
          <User className="w-3 h-3" /> Click a player to view heatmap
        </p>
        <div className="grid grid-cols-6 gap-1.5">
          {playerIds.slice(0, 24).map((pid, idx) => {
            const isTeamA = idx % 2 === 0;
            return (
              <button
                key={pid}
                onClick={() => onPlayerClick(pid)}
                className={`group relative p-2 rounded-lg text-center transition-all ${
                  isTeamA
                    ? 'bg-red-500/20 hover:bg-red-500/40 border border-red-500/30 hover:border-red-500/60'
                    : 'bg-blue-500/20 hover:bg-blue-500/40 border border-blue-500/30 hover:border-blue-500/60'
                }`}
              >
                <span className={`text-xs font-mono font-bold ${isTeamA ? 'text-red-300' : 'text-blue-300'}`}>{pid}</span>
                <Flame className="w-2.5 h-2.5 absolute top-0.5 right-0.5 text-orange-400 opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>
            );
          })}
        </div>
        {playerIds.length > 24 && (
          <p className="text-xs text-surface-500 text-center">+{playerIds.length - 24} more</p>
        )}
      </div>
    </div>
  );
}

// ============================================================
// HEATMAP MODAL
// ============================================================

function HeatmapModal({ videoId, playerId, onClose }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true); setError(null);
      try {
        const res = await fetch(`${API_BASE_URL}/heatmap/${videoId}/${playerId}`);
        if (!res.ok) throw new Error('Failed to load heatmap');
        setData(await res.json());
      } catch (e) { setError(e.message); }
      finally { setLoading(false); }
    };
    if (videoId && playerId !== null) load();
  }, [videoId, playerId]);

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-surface-900 border border-surface-700 rounded-2xl max-w-2xl w-full p-6 relative shadow-2xl">
        <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-lg hover:bg-surface-700 transition-colors">
          <X className="w-5 h-5 text-surface-400" />
        </button>
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 rounded-xl bg-gradient-to-br from-orange-500/20 to-red-500/20 border border-orange-500/30">
            <Flame className="w-6 h-6 text-orange-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Player #{playerId} Heatmap</h3>
            <p className="text-sm text-surface-400">Position frequency analysis</p>
          </div>
        </div>

        {loading && (
          <div className="aspect-[7/4.5] flex items-center justify-center bg-surface-800 rounded-xl">
            <Loader2 className="w-8 h-8 text-pitch-400 animate-spin" />
          </div>
        )}
        {error && (
          <div className="aspect-[7/4.5] flex items-center justify-center bg-surface-800 rounded-xl">
            <div className="text-center">
              <AlertCircle className="w-8 h-8 text-red-400 mx-auto mb-2" />
              <p className="text-sm text-red-300">{error}</p>
            </div>
          </div>
        )}
        {data && (
          <>
            <div className="rounded-xl overflow-hidden border border-surface-700 mb-4">
              <img src={`data:image/jpeg;base64,${data.heatmap}`} alt={`Heatmap #${playerId}`} className="w-full h-auto" />
            </div>
            {data.stats && (
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: 'Positions',    val: data.stats.position_count ?? 0,               fmt: (v) => v },
                  { label: 'Avg. X',       val: data.stats.avg_x ?? 0,                        fmt: (v) => `${v.toFixed(0)}px` },
                  { label: 'Avg. Y',       val: data.stats.avg_y ?? 0,                        fmt: (v) => `${v.toFixed(0)}px` },
                ].map(({ label, val, fmt }) => (
                  <div key={label} className="bg-surface-800/50 rounded-lg p-3 text-center border border-surface-700">
                    <p className="text-xs text-surface-400 mb-1">{label}</p>
                    <p className="text-lg font-bold text-pitch-400">{fmt(val)}</p>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ============================================================
// SETTINGS PANEL
// ============================================================

function SettingsPanel({ options, onOptionsChange }) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider flex items-center gap-2">
        <Settings className="w-4 h-4" /> Display Options
      </h3>
      <div className="space-y-3">
        <ToggleSwitch enabled={options.showTracking}  onChange={(v) => onOptionsChange({ ...options, showTracking: v })}  label="Player Tracks"  icon={Footprints} />
        <ToggleSwitch enabled={options.showVoronoi}   onChange={(v) => onOptionsChange({ ...options, showVoronoi: v })}   label="Voronoi Zones"  icon={Hexagon} />
        <ToggleSwitch enabled={options.showRadar}     onChange={(v) => onOptionsChange({ ...options, showRadar: v })}     label="Radar View"     icon={Radar} />
        <ToggleSwitch enabled={options.showPlayerIds} onChange={(v) => onOptionsChange({ ...options, showPlayerIds: v })} label="Player IDs"     icon={Hash} />
        <ToggleSwitch enabled={options.showSpeeds}    onChange={(v) => onOptionsChange({ ...options, showSpeeds: v })}    label="Speed Overlay"  icon={Gauge} />
      </div>
    </div>
  );
}

// ============================================================
// STATS DASHBOARD
// ============================================================

function StatsDashboard({ stats }) {
  const {
    players_detected = 0,
    referees_detected = 0,
    ball_detected = false,
    team_a_count = 0,
    team_b_count = 0,
    possession_team_a = 50,
    possession_team_b = 50,
    frame_number = 0,
    is_calibrated = false,
    homography_auto = false,
  } = stats;

  return (
    <div className="space-y-6">
      {/* Possession */}
      <div>
        <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider mb-3 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" /> Spatial Control
        </h3>
        <PossessionBar teamA={possession_team_a} teamB={possession_team_b} />
      </div>

      {/* Team Counts */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Team A" value={team_a_count} icon={Users} color="red" />
        <StatCard label="Team B" value={team_b_count} icon={Users} color="blue" />
      </div>

      {/* Detection Stats */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Players" value={players_detected} icon={Users} color="pitch" subValue={`${referees_detected} refs filtered`} />
        <StatCard label="Ball" value={ball_detected ? 'Tracked' : 'Lost'} icon={Target} color={ball_detected ? 'pitch' : 'yellow'} />
      </div>

      {/* Frame / System Info */}
      <div className="pt-3 border-t border-surface-700/50 space-y-1.5">
        <div className="flex justify-between text-xs">
          <span className="text-surface-500">Frame</span>
          <span className="font-mono text-surface-400">{frame_number}</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-surface-500">Team Calibration</span>
          <span className={is_calibrated ? 'text-pitch-400' : 'text-yellow-400'}>
            {is_calibrated ? 'Complete' : 'Learning...'}
          </span>
        </div>
        {/* NEW: Auto homography badge */}
        <div className="flex justify-between text-xs">
          <span className="text-surface-500">Homography</span>
          <span className={`flex items-center gap-1 ${homography_auto ? 'text-pitch-400' : 'text-yellow-400'}`}>
            {homography_auto ? (
              <><Check className="w-3 h-3" /> Auto keypoints</>
            ) : (
              <><AlertCircle className="w-3 h-3" /> Fallback mode</>
            )}
          </span>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// HEADER
// ============================================================

function Header() {
  return (
    <header className="border-b border-surface-800 bg-surface-950/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="max-w-[1800px] mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pitch-500 to-pitch-700 flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-3 h-3 rounded-full bg-pitch-400 border-2 border-surface-950 animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg font-bold font-display text-white">Football AI Analyzer</h1>
              <p className="text-xs text-surface-500">Tactical Vision System v3.0</p>
            </div>
          </div>
          <span className="flex items-center gap-2 text-xs text-surface-500 bg-surface-800/50 px-3 py-1.5 rounded-full">
            <span className="w-2 h-2 rounded-full bg-pitch-500 animate-pulse" />
            CV Pipeline Active
          </span>
        </div>
      </div>
    </header>
  );
}

// ============================================================
// MAIN APP
// ============================================================

export default function App() {
  const [videoId,       setVideoId]       = useState(null);
  const [status,        setStatus]        = useState('idle');
  const [progress,      setProgress]      = useState(0);
  const [frameData,     setFrameData]     = useState(null);
  const [radarData,     setRadarData]     = useState(null);
  const [stats,         setStats]         = useState({});
  const [errorMessage,  setErrorMessage]  = useState(null);
  const [sprintAlerts,  setSprintAlerts]  = useState([]);

  const [options, setOptions] = useState({
    showTracking:  true,
    showVoronoi:   true,
    showRadar:     true,
    showPlayerIds: true,
    showSpeeds:    true,
  });

  const [selectedPlayerId, setSelectedPlayerId] = useState(null);
  const [showHeatmap,      setShowHeatmap]      = useState(false);

  // Auto-dismiss sprint alerts after 4 seconds
  useEffect(() => {
    if (sprintAlerts.length === 0) return;
    const t = setTimeout(() => setSprintAlerts([]), 4000);
    return () => clearTimeout(t);
  }, [sprintAlerts]);

  const handleWsMessage = useCallback((data) => {
    switch (data.type) {
      case 'frame':
        setFrameData(data.frame);
        setRadarData(data.radar);
        setStats(data.stats || {});
        setProgress(data.progress || 0);
        // Accumulate sprint alerts
        if (data.stats?.sprint_alerts?.length > 0) {
          setSprintAlerts(prev => [...prev, ...data.stats.sprint_alerts].slice(-5));
        }
        break;
      case 'complete':
        setStatus('completed');
        setProgress(100);
        break;
      case 'error':
        setStatus('error');
        setErrorMessage(data.message);
        break;
      case 'status':
        if (data.status)   setStatus(data.status);
        if (data.progress) setProgress(data.progress);
        break;
    }
  }, []);

  useWebSocket(status === 'processing' ? videoId : null, handleWsMessage);

  const handleUpload = async (file) => {
    setStatus('uploading');
    setErrorMessage(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const upRes = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: fd });
      if (!upRes.ok) throw new Error('Upload failed');
      const upData = await upRes.json();
      setVideoId(upData.video_id);

      const prRes = await fetch(`${API_BASE_URL}/process/${upData.video_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          show_tracking: options.showTracking,
          show_voronoi:  options.showVoronoi,
          show_radar:    options.showRadar,
          target_fps:    15,
        }),
      });
      if (!prRes.ok) throw new Error('Failed to start processing');
      setStatus('processing');
    } catch (e) {
      setStatus('error');
      setErrorMessage(e.message);
    }
  };

  const handleReset = () => {
    setVideoId(null); setStatus('idle'); setProgress(0);
    setFrameData(null); setRadarData(null); setStats({});
    setErrorMessage(null); setSprintAlerts([]);
  };

  const playerSpeeds    = stats.player_speeds    ?? {};
  const playerDistances = stats.player_distances ?? {};
  const teamStats       = stats.team_stats       ?? {};
  const topSpeeds       = stats.top_speeds       ?? [];
  const activeIds       = stats.active_player_ids ?? [];

  return (
    <div className="min-h-screen bg-surface-950">
      <Header />

      <main className="max-w-[1800px] mx-auto px-6 py-6">
        <div className="grid grid-cols-12 gap-6">

          {/* ── LEFT SIDEBAR ── */}
          <aside className="col-span-12 lg:col-span-3 xl:col-span-2">
            <div className="bg-surface-900/50 backdrop-blur-sm rounded-2xl border border-surface-800 p-5 space-y-6 sticky top-24">
              <SettingsPanel options={options} onOptionsChange={setOptions} />

              {status !== 'idle' && (
                <div className="pt-4 border-t border-surface-700/50">
                  <ProgressBar progress={progress} status={status} />
                </div>
              )}

              <div className="space-y-2">
                {status === 'completed' && (
                  <button
                    onClick={() => window.open(`${API_BASE_URL}/output/${videoId}`, '_blank')}
                    className="w-full py-2.5 px-4 bg-pitch-600 hover:bg-pitch-500 text-white rounded-xl font-medium flex items-center justify-center gap-2 transition-colors"
                  >
                    <Download className="w-4 h-4" /> Download Video
                  </button>
                )}
                {status !== 'idle' && (
                  <button onClick={handleReset} className="w-full py-2.5 px-4 bg-surface-700 hover:bg-surface-600 text-surface-200 rounded-xl font-medium transition-colors">
                    New Analysis
                  </button>
                )}
              </div>

              {errorMessage && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
                    <p className="text-sm text-red-300">{errorMessage}</p>
                  </div>
                </div>
              )}
            </div>
          </aside>

          {/* ── MAIN CONTENT ── */}
          <section className="col-span-12 lg:col-span-6 xl:col-span-7 space-y-4">
            {status === 'idle' ? (
              <VideoDropzone onUpload={handleUpload} isUploading={status === 'uploading'} />
            ) : (
              <>
                <div className="bg-surface-900/50 backdrop-blur-sm rounded-2xl border border-surface-800 p-4">
                  <VideoPlayer frameData={frameData} isProcessing={status === 'processing' || status === 'uploading'} />
                </div>

                {/* Sprint alert */}
                {sprintAlerts.length > 0 && (
                  <SprintAlertToast alerts={sprintAlerts} onDismiss={() => setSprintAlerts([])} />
                )}

                {status === 'processing' && (
                  <div className="flex items-center justify-between p-3 rounded-xl bg-pitch-900/30 border border-pitch-700/30">
                    <div className="flex items-center gap-3">
                      <Loader2 className="w-4 h-4 text-pitch-400 animate-spin" />
                      <span className="text-sm text-pitch-300">Analyzing... {progress.toFixed(0)}% complete</span>
                    </div>
                    <span className="text-xs text-surface-500 font-mono">Frame {stats.frame_number ?? 0}</span>
                  </div>
                )}

                {status === 'completed' && (
                  <div className="flex items-center gap-3 p-3 rounded-xl bg-pitch-900/30 border border-pitch-700/30">
                    <Check className="w-4 h-4 text-pitch-400" />
                    <span className="text-sm text-pitch-300">Analysis complete! Download video or view player heatmaps.</span>
                  </div>
                )}

                {/* ── KINEMATICS PANELS (below video) ── */}
                {Object.keys(teamStats).length > 0 && (
                  <div className="grid grid-cols-2 gap-4">
                    {/* Team Distance */}
                    <div className="bg-surface-900/50 backdrop-blur-sm rounded-2xl border border-surface-800 p-4 space-y-3">
                      <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider flex items-center gap-2">
                        <Route className="w-4 h-4 text-pitch-400" /> Distance Covered
                      </h3>
                      <TeamDistancePanel teamStats={teamStats} />
                    </div>

                    {/* Top Speeds */}
                    <div className="bg-surface-900/50 backdrop-blur-sm rounded-2xl border border-surface-800 p-4 space-y-3">
                      <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-pitch-400" /> Peak Speeds
                      </h3>
                      <TopSpeedsLeaderboard topSpeeds={topSpeeds} />
                    </div>
                  </div>
                )}
              </>
            )}
          </section>

          {/* ── RIGHT SIDEBAR ── */}
          <aside className="col-span-12 lg:col-span-3 space-y-4">
            {/* Player Radar */}
            {options.showRadar && (
              <div className="bg-surface-900/50 backdrop-blur-sm rounded-2xl border border-surface-800 p-4">
                <PlayerRadar radarData={radarData} stats={stats} onPlayerClick={(id) => { setSelectedPlayerId(id); setShowHeatmap(true); }} />
              </div>
            )}

            {/* Live Speed Ticker */}
            {options.showSpeeds && (
              <div className="bg-surface-900/50 backdrop-blur-sm rounded-2xl border border-surface-800 p-4 space-y-3">
                <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-pitch-400" /> Live Speeds
                </h3>
                <SpeedTicker playerSpeeds={playerSpeeds} trackedIds={activeIds} />
              </div>
            )}

            {/* Stats Panel */}
            <div className="bg-surface-900/50 backdrop-blur-sm rounded-2xl border border-surface-800 p-5">
              <StatsDashboard stats={stats} />
            </div>

            {/* Pipeline Info */}
            <div className="bg-surface-900/50 backdrop-blur-sm rounded-2xl border border-surface-800 p-4">
              <h4 className="text-sm font-semibold text-surface-300 mb-3 flex items-center gap-2">
                <Zap className="w-4 h-4 text-pitch-400" /> Pipeline v3.0
              </h4>
              <ul className="space-y-2 text-xs text-surface-400">
                {[
                  { text: 'Auto pitch keypoint detection',   color: 'text-pitch-500' },
                  { text: 'RANSAC homography estimation',     color: 'text-pitch-500' },
                  { text: 'Real-world speed & distance',      color: 'text-pitch-500' },
                  { text: 'Referee & boundary filtering',     color: 'text-pitch-500' },
                  { text: 'Kalman-smoothed tracking',         color: 'text-pitch-500' },
                  { text: 'Persistent unique player IDs',     color: 'text-pitch-500' },
                  { text: 'Individual player heatmaps',       color: 'text-orange-400' },
                ].map(({ text, color }) => (
                  <li key={text} className="flex items-center gap-2">
                    <ChevronRight className={`w-3 h-3 ${color}`} />
                    {text}
                  </li>
                ))}
              </ul>
            </div>
          </aside>
        </div>
      </main>

      {/* Heatmap Modal */}
      {showHeatmap && selectedPlayerId !== null && videoId && (
        <HeatmapModal
          videoId={videoId}
          playerId={selectedPlayerId}
          onClose={() => { setShowHeatmap(false); setSelectedPlayerId(null); }}
        />
      )}

      <footer className="border-t border-surface-800 mt-12">
        <div className="max-w-[1800px] mx-auto px-6 py-4">
          <p className="text-center text-xs text-surface-600">
            Football AI Analyzer v3.0 · Auto Homography · Player Kinematics · Heatmaps · Voronoi
          </p>
        </div>
      </footer>
    </div>
  );
}