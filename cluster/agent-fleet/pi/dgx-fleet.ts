import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
export default function (pi: ExtensionAPI) {
  const restrictWorkerTools = () => {
    const permitted = new Set(['read', 'bash', 'edit', 'write', 'grep', 'find', 'ls']);
    // Intersect the profile's existing tools; never add write access to a
    // read-only worker. Approval extensions remain loaded and active.
    const tools = pi.getActiveTools().filter(name => permitted.has(name));
    pi.setActiveTools(tools);
    if (process.env.DGX_FLEET_TRACE === '1') {
      process.stderr.write(JSON.stringify({event: 'dgx_worker_tools', tools}) + '\n');
    }
  };
  pi.on('session_start', (_event, ctx) => {
    if (ctx.model?.provider === 'dgx-spark' && ctx.model.id.startsWith('qwen')) restrictWorkerTools();
  });
  pi.on('before_provider_request', (event, ctx) => {
    if (ctx.model?.provider !== 'dgx-spark') return;
    const payload = { ...(event.payload as Record<string, unknown>) };
    if (ctx.model.id.startsWith('qwen')) {
      payload.chat_template_kwargs = { ...((payload.chat_template_kwargs as object) || {}), enable_thinking: false };
      delete payload.reasoning_effort;
      payload.priority = 20;
    } else {
      const effort = payload.reasoning_effort ?? pi.getThinkingLevel();
      payload.reasoning_effort = effort === 'low' ? 'low' : effort === 'max' || effort === 'xhigh' ? 'max' : 'high';
      payload.priority = 0;
    }
    if (process.env.DGX_FLEET_TRACE === '1') {
      process.stderr.write(JSON.stringify({ event: 'dgx_fleet_route', model: ctx.model.id,
        reasoning_effort: payload.reasoning_effort, chat_template_kwargs: payload.chat_template_kwargs,
        priority: payload.priority }) + '\n');
    }
    return payload;
  });
  pi.on('before_agent_start', (event, ctx) => {
    if (ctx.model?.provider !== 'dgx-spark') return;
    if (ctx.model.id.startsWith('qwen')) { restrictWorkerTools(); return; }
    if (ctx.model.id !== 'dgx-orchestrator') {
      return { systemPrompt: event.systemPrompt + '\n\nThis recipe uses one shared model engine. There is no separately reserved Qwen worker pool. Do simple work directly, keep delegation bounded, preserve approvals, and own integration. Do not infer extra capacity from agent names.' };
    }
    return { systemPrompt: event.systemPrompt + `\n\nLocal Spark fleet: you are the GLM coordinator on two dedicated Sparks. Qwen workers have two separate Sparks. For substantial work with independent branches, delegate bounded investigation or implementation to the installed Explore, general-purpose, dgx-worker-a, or dgx-worker-b agents. Give each child a concise exact scope, relevant paths, and acceptance criteria. Reference contracts already in files instead of repeating them. Routing is preconfigured; do not audit runtime configuration as part of an ordinary task. Use explicit worker A/B types to preserve replica affinity for long iterative work; the generic worker pool spills simultaneous requests across replicas. Use joined/foreground Agent calls (run_in_background: false) for batch or headless work, placing independent calls in the same turn to run concurrently. A final answer terminates a headless session and cancels background children; do not end with a promise to wait for notifications. Keep useful independent work locally. Assign disjoint write paths, preserve approval enforcement, join results, inspect evidence, and own integration and final verification. Prefer 2-4 useful simultaneous workers. Use Qwen for routine bounded modules and factual investigation. Keep strict protocol parsers, difficult state machines, disputed correctness, and integration on GLM. If a child exhausts its budget, inspect its partial result and resolve the hard part locally; do not repeatedly reassign the same failed repair. Include scope and stop-on-denial constraints in child briefs. Do simple tasks directly. Do not fan out dependent work or send the whole history when a focused brief suffices. Plan and dgx-reviewer use GLM for difficult design/review. Never claim delegated work passed without checking its result. Validate edge cases explicitly stated in the contract, not only visible smoke tests. After edits, run tests in a fresh process to avoid stale imports in persistent kernels. Once relevant checks pass and no new change or concern exists, finish; repeated unchanged testing is not progress.` };
  });
}
