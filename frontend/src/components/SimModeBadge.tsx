import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";

/**
 * Persistent corner badge shown when the backend runs with simulated hardware
 * (`lelab --sim`). Without it, sine-wave joint data and synthetic camera frames
 * are easy to mistake for a real arm — every page can produce data, and none of
 * it comes from a robot.
 */
const SimModeBadge = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [sim, setSim] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/health`);
        if (!response.ok) return;
        const data = await response.json();
        if (!cancelled) setSim(Boolean(data.sim_mode));
      } catch {
        // Backend unreachable: other surfaces already report that.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  if (!sim) return null;

  return (
    <div
      role="status"
      title="Simulation mode: arms and cameras are faked, no robot is connected"
      className="fixed bottom-3 left-3 z-50 inline-flex items-center gap-2 rounded-full border border-amber-500/40 bg-amber-500/15 px-3 py-1 text-xs font-bold tracking-widest text-amber-300 backdrop-blur"
    >
      <FlaskConical className="h-3.5 w-3.5" />
      SIMULATION
    </div>
  );
};

export default SimModeBadge;
