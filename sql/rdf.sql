-- Context re-derivation frequency (RDF) and context-assembly cost share (CACS), on DuckDB.
--
-- RDF  = re-sent input tokens / all input tokens, per task: the share of what an agent sends
--        to the model that it had already sent earlier in the same task.
-- CACS = the share of a task's spend that re-sent input costs. Here it is MODELED from token
--        counts under the price ratios in `params`, not taken from provider bills.
--
-- Inputs (views created by rdf_tier0.py):
--   messages (traj_id BIGINT, idx INTEGER, role VARCHAR, tokens BIGINT)
--       one row per message of a trajectory, in order
--   task_meta(traj_id BIGINT, trajectory_id VARCHAR, instance_id VARCHAR, repo VARCHAR,
--             resolved BIGINT, exit_status VARCHAR, tools_tokens BIGINT)
--       one row per trajectory; tools_tokens = size of the tool definitions sent with every call
--
-- Replay model: an append-only loop, which is how the source runs were configured (OpenHands
-- no-op condenser, history truncation off, native tool calling). Each model call - one assistant
-- message - receives the tool definitions plus every message before it. A token is NEW input the
-- first time a call sends it and RE-SENT input on every later call. Messages after the last
-- assistant message are never sent to the model, so they count nowhere.

-- 0. Price ratios for the modeled CACS, relative to a fresh input token = 1.0.
CREATE OR REPLACE TABLE params AS
SELECT 5.0::DOUBLE  AS p_output,       -- assumed output price
       0.10::DOUBLE AS p_cache_read,   -- re-sent input under prefix caching
       1.25::DOUBLE AS p_cache_write;  -- new input written to the cache

-- 1. One row per model call, with the size of the message history it received.
CREATE OR REPLACE TABLE calls AS
SELECT traj_id, idx, history_tokens
FROM (
    SELECT traj_id,
           idx,
           role,
           coalesce(sum(tokens) OVER (PARTITION BY traj_id ORDER BY idx
                                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)
               AS history_tokens
    FROM messages
)
WHERE role = 'assistant';

-- 2. One row per task.
CREATE OR REPLACE TABLE tasks AS
WITH per_task AS (
    SELECT traj_id,
           count(*)            AS n_calls,
           sum(history_tokens) AS history_input,  -- message tokens summed over every call
           max(history_tokens) AS history_new,    -- each message is new once; the last call holds them all
           min(history_tokens) AS first_history   -- system prompt + task statement, before the first call
    FROM calls
    GROUP BY traj_id
),
outputs AS (
    SELECT traj_id,
           coalesce(sum(tokens) FILTER (WHERE role = 'assistant'), 0) AS output_tokens
    FROM messages
    GROUP BY traj_id
),
t AS (
    SELECT m.traj_id, m.trajectory_id, m.instance_id, m.repo, m.resolved, m.exit_status,
           p.n_calls,
           m.tools_tokens,
           p.history_input + p.n_calls * m.tools_tokens AS input_tokens,
           p.history_new + m.tools_tokens               AS new_input_tokens,
           p.first_history + m.tools_tokens             AS preamble_tokens,
           o.output_tokens,
           p.history_input,
           p.history_new
    FROM per_task p
    JOIN task_meta m USING (traj_id)
    JOIN outputs o USING (traj_id)
)
SELECT traj_id, trajectory_id, instance_id, repo, resolved, exit_status,
       n_calls,
       input_tokens,
       new_input_tokens,
       input_tokens - new_input_tokens                                      AS resent_input_tokens,
       output_tokens,
       tools_tokens,
       preamble_tokens,
       (n_calls - 1) * preamble_tokens                                      AS preamble_resent_tokens,

       -- the metric
       (input_tokens - new_input_tokens)::DOUBLE / input_tokens             AS rdf,
       -- sensitivity check: messages only, tool definitions left out
       (history_input - history_new)::DOUBLE / nullif(history_input, 0)     AS rdf_messages_only,
       -- the thesis's closed form for a loop that adds equal new context per call
       1 - 2.0 / (n_calls + 1)                                              AS rdf_closed_form,
       -- share of re-sent tokens that is the fixed preamble (tools + system prompt + task)
       ((n_calls - 1) * preamble_tokens)::DOUBLE
           / nullif(input_tokens - new_input_tokens, 0)                     AS preamble_share_of_resent,

       -- modeled CACS, no caching: every input token at the fresh-input price
       (input_tokens - new_input_tokens)
           / (input_tokens + p_output * output_tokens)                      AS cacs_no_cache,
       -- modeled CACS, perfect prefix caching: re-sent tokens at the cache-read price
       p_cache_read * (input_tokens - new_input_tokens)
           / (  p_cache_read * (input_tokens - new_input_tokens)
              + p_cache_write * new_input_tokens
              + p_output * output_tokens)                                   AS cacs_prefix_cache
FROM t, params;

-- 3. Summaries.
CREATE OR REPLACE TABLE summary_overall AS
SELECT count(*)                                                    AS tasks,
       sum(resolved)                                               AS resolved_tasks,
       quantile_cont(n_calls, 0.5)                                 AS calls_median,
       sum(input_tokens)                                           AS input_tokens_total,
       sum(resent_input_tokens)                                    AS resent_tokens_total,
       sum(resent_input_tokens)::DOUBLE / sum(input_tokens)        AS rdf_pooled,
       quantile_cont(rdf, 0.5)                                     AS rdf_median,
       quantile_cont(rdf, 0.1)                                     AS rdf_p10,
       quantile_cont(rdf, 0.9)                                     AS rdf_p90,
       quantile_cont(rdf_messages_only, 0.5)                       AS rdf_messages_only_median,
       quantile_cont(rdf - rdf_closed_form, 0.5)                   AS gap_vs_closed_form_median,
       quantile_cont(input_tokens, 0.5)                            AS input_tokens_median,
       sum(preamble_resent_tokens)::DOUBLE
           / sum(resent_input_tokens)                              AS preamble_share_pooled,
       sum(preamble_resent_tokens)::DOUBLE / sum(input_tokens)     AS preamble_resent_share_of_input,
       sum(resent_input_tokens)
           / sum(input_tokens + p_output * output_tokens)          AS cacs_no_cache_pooled,
       sum(p_cache_read * resent_input_tokens)
           / sum(  p_cache_read * resent_input_tokens
                 + p_cache_write * new_input_tokens
                 + p_output * output_tokens)                       AS cacs_prefix_cache_pooled,
       quantile_cont(cacs_no_cache, 0.5)                           AS cacs_no_cache_median,
       quantile_cont(cacs_prefix_cache, 0.5)                       AS cacs_prefix_cache_median
FROM tasks, params;

CREATE OR REPLACE TABLE summary_by_outcome AS
SELECT CASE WHEN resolved = 1 THEN 'resolved' ELSE 'not resolved' END AS outcome,
       count(*)                                                    AS tasks,
       quantile_cont(n_calls, 0.5)                                 AS calls_median,
       quantile_cont(rdf, 0.5)                                     AS rdf_median,
       quantile_cont(rdf, 0.1)                                     AS rdf_p10,
       quantile_cont(rdf, 0.9)                                     AS rdf_p90,
       quantile_cont(input_tokens, 0.5)                            AS input_tokens_median
FROM tasks
GROUP BY ALL
ORDER BY outcome DESC;

CREATE OR REPLACE TABLE rdf_by_calls AS
SELECT n_calls,
       count(*)                    AS tasks,
       quantile_cont(rdf, 0.1)     AS p10,
       quantile_cont(rdf, 0.5)     AS p50,
       quantile_cont(rdf, 0.9)     AS p90,
       any_value(rdf_closed_form)  AS closed_form
FROM tasks
GROUP BY n_calls
ORDER BY n_calls;

CREATE OR REPLACE TABLE summary_by_band AS
SELECT CASE WHEN n_calls <= 10 THEN '1-10'
            WHEN n_calls <= 25 THEN '11-25'
            WHEN n_calls <= 50 THEN '26-50'
            WHEN n_calls <= 75 THEN '51-75'
            ELSE '76+' END                                         AS calls_band,
       min(n_calls)                                                AS band_start,
       count(*)                                                    AS tasks,
       quantile_cont(rdf, 0.5)                                     AS rdf_median,
       quantile_cont(rdf, 0.1)                                     AS rdf_p10,
       quantile_cont(rdf, 0.9)                                     AS rdf_p90,
       quantile_cont(rdf_closed_form, 0.5)                         AS closed_form_median
FROM tasks
GROUP BY calls_band
ORDER BY band_start;
