-- Azure SQL schema for the sustainability advisor.
-- Mirrors src/sustainability_advisor/db/tables.py. Apply with sqlcmd or the
-- CD pipeline. Idempotent: every object is created only when missing.

IF SCHEMA_ID(N'dbo') IS NULL EXEC(N'CREATE SCHEMA dbo');
GO

IF OBJECT_ID(N'dbo.facilities', N'U') IS NULL
CREATE TABLE dbo.facilities (
    facility_id    NVARCHAR(64)  NOT NULL CONSTRAINT pk_facilities PRIMARY KEY,
    name           NVARCHAR(200) NOT NULL,
    region         NVARCHAR(100) NULL,
    country        NVARCHAR(100) NULL,
    industry       NVARCHAR(100) NULL,
    capacity       FLOAT         NULL,
    capacity_unit  NVARCHAR(50)  NULL,
    is_active      BIT           NOT NULL CONSTRAINT df_facilities_active DEFAULT (1)
);
GO

IF OBJECT_ID(N'dbo.emissions', N'U') IS NULL
CREATE TABLE dbo.emissions (
    emission_id   INT IDENTITY(1,1) NOT NULL CONSTRAINT pk_emissions PRIMARY KEY,
    facility_id   NVARCHAR(64)  NOT NULL CONSTRAINT fk_emissions_facility REFERENCES dbo.facilities (facility_id),
    period_start  DATE          NOT NULL,
    period_end    DATE          NOT NULL,
    scope         INT           NOT NULL CONSTRAINT ck_emissions_scope CHECK (scope IN (1, 2, 3)),
    category      NVARCHAR(100) NULL,
    co2e_tonnes   FLOAT         NOT NULL,
    source        NVARCHAR(100) NULL,
    methodology   NVARCHAR(100) NULL
);
CREATE INDEX ix_emissions_facility_period ON dbo.emissions (facility_id, period_start);
GO

IF OBJECT_ID(N'dbo.energy', N'U') IS NULL
CREATE TABLE dbo.energy (
    energy_id        INT IDENTITY(1,1) NOT NULL CONSTRAINT pk_energy PRIMARY KEY,
    facility_id      NVARCHAR(64) NOT NULL CONSTRAINT fk_energy_facility REFERENCES dbo.facilities (facility_id),
    period_start     DATE         NOT NULL,
    period_end       DATE         NOT NULL,
    energy_type      NVARCHAR(50) NOT NULL,
    consumption_mwh  FLOAT        NOT NULL,
    renewable_mwh    FLOAT        NOT NULL,
    cost             FLOAT        NULL,
    currency         NCHAR(3)     NULL
);
CREATE INDEX ix_energy_facility_period ON dbo.energy (facility_id, period_start);
GO

IF OBJECT_ID(N'dbo.production', N'U') IS NULL
CREATE TABLE dbo.production (
    production_id    INT IDENTITY(1,1) NOT NULL CONSTRAINT pk_production PRIMARY KEY,
    facility_id      NVARCHAR(64)  NOT NULL CONSTRAINT fk_production_facility REFERENCES dbo.facilities (facility_id),
    period_start     DATE          NOT NULL,
    period_end       DATE          NOT NULL,
    product          NVARCHAR(100) NULL,
    output_quantity  FLOAT         NOT NULL,
    output_unit      NVARCHAR(50)  NULL
);
CREATE INDEX ix_production_facility_period ON dbo.production (facility_id, period_start);
GO

IF OBJECT_ID(N'dbo.sensors', N'U') IS NULL
CREATE TABLE dbo.sensors (
    sensor_id    NVARCHAR(64)  NOT NULL CONSTRAINT pk_sensors PRIMARY KEY,
    facility_id  NVARCHAR(64)  NOT NULL CONSTRAINT fk_sensors_facility REFERENCES dbo.facilities (facility_id),
    sensor_type  NVARCHAR(100) NOT NULL,
    unit         NVARCHAR(50)  NULL,
    location     NVARCHAR(200) NULL
);
GO

IF OBJECT_ID(N'dbo.sensor_readings', N'U') IS NULL
CREATE TABLE dbo.sensor_readings (
    reading_id   INT IDENTITY(1,1) NOT NULL CONSTRAINT pk_sensor_readings PRIMARY KEY,
    sensor_id    NVARCHAR(64) NOT NULL CONSTRAINT fk_readings_sensor REFERENCES dbo.sensors (sensor_id),
    recorded_at  DATETIME2    NOT NULL,
    value        FLOAT        NOT NULL
);
CREATE INDEX ix_sensor_readings_sensor_time ON dbo.sensor_readings (sensor_id, recorded_at);
GO

IF OBJECT_ID(N'dbo.sustainability_metrics', N'U') IS NULL
CREATE TABLE dbo.sustainability_metrics (
    metric_id     INT IDENTITY(1,1) NOT NULL CONSTRAINT pk_sustainability_metrics PRIMARY KEY,
    facility_id   NVARCHAR(64)  NOT NULL CONSTRAINT fk_metrics_facility REFERENCES dbo.facilities (facility_id),
    metric_name   NVARCHAR(100) NOT NULL,
    period_start  DATE          NOT NULL,
    period_end    DATE          NOT NULL,
    value         FLOAT         NULL,
    unit          NVARCHAR(50)  NULL,
    target_value  FLOAT         NULL,
    source        NVARCHAR(100) NULL
);
CREATE INDEX ix_metrics_facility_name ON dbo.sustainability_metrics (facility_id, metric_name, period_start);
GO

IF OBJECT_ID(N'dbo.anomalies', N'U') IS NULL
CREATE TABLE dbo.anomalies (
    anomaly_id      NVARCHAR(64)  NOT NULL CONSTRAINT pk_anomalies PRIMARY KEY,
    facility_id     NVARCHAR(64)  NOT NULL CONSTRAINT fk_anomalies_facility REFERENCES dbo.facilities (facility_id),
    metric          NVARCHAR(100) NOT NULL,
    period_start    DATE          NOT NULL,
    observed_value  FLOAT         NOT NULL,
    expected_value  FLOAT         NOT NULL,
    score           FLOAT         NOT NULL,
    severity        NVARCHAR(20)  NOT NULL,
    method          NVARCHAR(50)  NOT NULL,
    detected_at     DATETIME2     NOT NULL,
    status          NVARCHAR(20)  NOT NULL
);
GO

IF OBJECT_ID(N'dbo.recommendations', N'U') IS NULL
CREATE TABLE dbo.recommendations (
    recommendation_id  NVARCHAR(64)  NOT NULL CONSTRAINT pk_recommendations PRIMARY KEY,
    facility_id        NVARCHAR(64)  NULL,
    conversation_id    NVARCHAR(64)  NULL,
    issue              NVARCHAR(MAX) NOT NULL,
    severity           NVARCHAR(20)  NOT NULL,
    priority           NVARCHAR(20)  NOT NULL,
    confidence         FLOAT         NOT NULL,
    payload            NVARCHAR(MAX) NOT NULL CONSTRAINT ck_recommendations_json CHECK (ISJSON(payload) = 1),
    created_at         DATETIME2     NOT NULL
);
CREATE INDEX ix_recommendations_conversation ON dbo.recommendations (conversation_id, created_at);
GO

IF OBJECT_ID(N'dbo.forecast_results', N'U') IS NULL
CREATE TABLE dbo.forecast_results (
    forecast_id    NVARCHAR(64)  NOT NULL CONSTRAINT pk_forecast_results PRIMARY KEY,
    facility_id    NVARCHAR(64)  NOT NULL,
    metric         NVARCHAR(100) NOT NULL,
    method         NVARCHAR(50)  NOT NULL,
    horizon        INT           NOT NULL,
    backtest_mape  FLOAT         NULL,
    payload        NVARCHAR(MAX) NOT NULL CONSTRAINT ck_forecast_json CHECK (ISJSON(payload) = 1),
    generated_at   DATETIME2     NOT NULL
);
CREATE INDEX ix_forecast_facility_metric ON dbo.forecast_results (facility_id, metric, generated_at);
GO

IF OBJECT_ID(N'dbo.documents', N'U') IS NULL
CREATE TABLE dbo.documents (
    document_id     NVARCHAR(128)  NOT NULL CONSTRAINT pk_documents PRIMARY KEY,
    title           NVARCHAR(300)  NOT NULL,
    doc_type        NVARCHAR(100)  NOT NULL,
    source_uri      NVARCHAR(1000) NOT NULL,
    jurisdiction    NVARCHAR(100)  NULL,
    facility_id     NVARCHAR(64)   NULL,
    effective_date  DATE           NULL,
    version         NVARCHAR(50)   NULL,
    checksum        NVARCHAR(128)  NOT NULL,
    indexed_at      DATETIME2      NULL
);
GO

IF OBJECT_ID(N'dbo.audit_events', N'U') IS NULL
CREATE TABLE dbo.audit_events (
    event_id         NVARCHAR(64)  NOT NULL CONSTRAINT pk_audit_events PRIMARY KEY,
    conversation_id  NVARCHAR(64)  NULL,
    request_id       NVARCHAR(64)  NULL,
    event_type       NVARCHAR(100) NOT NULL,
    actor            NVARCHAR(100) NOT NULL,
    payload          NVARCHAR(MAX) NOT NULL CONSTRAINT ck_audit_json CHECK (ISJSON(payload) = 1),
    created_at       DATETIME2     NOT NULL
);
CREATE INDEX ix_audit_conversation ON dbo.audit_events (conversation_id, created_at);
GO

IF OBJECT_ID(N'dbo.conversations', N'U') IS NULL
CREATE TABLE dbo.conversations (
    conversation_id  NVARCHAR(64) NOT NULL CONSTRAINT pk_conversations PRIMARY KEY,
    created_at       DATETIME2    NOT NULL,
    updated_at       DATETIME2    NOT NULL
);
GO

IF OBJECT_ID(N'dbo.conversation_messages', N'U') IS NULL
CREATE TABLE dbo.conversation_messages (
    message_id       NVARCHAR(64)  NOT NULL CONSTRAINT pk_conversation_messages PRIMARY KEY,
    conversation_id  NVARCHAR(64)  NOT NULL CONSTRAINT fk_messages_conversation REFERENCES dbo.conversations (conversation_id),
    role             NVARCHAR(20)  NOT NULL,
    content          NVARCHAR(MAX) NOT NULL,
    payload          NVARCHAR(MAX) NOT NULL,
    created_at       DATETIME2     NOT NULL
);
CREATE INDEX ix_messages_conversation ON dbo.conversation_messages (conversation_id, created_at);
GO
