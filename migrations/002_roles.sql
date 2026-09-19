-- Least privilege roles for the two identities the service uses.
-- Replace the principal names with the managed identities of the deployment
-- (the CD pipeline passes them as sqlcmd variables).
--
-- $(APP_IDENTITY):   application engine; reads data, writes audit,
--                    conversation, recommendation, forecast, anomaly and
--                    document catalogue rows
-- $(QUERY_IDENTITY): read only engine for model generated SQL; can read only
--                    the tables exposed to the SQL agent

IF DATABASE_PRINCIPAL_ID(N'$(APP_IDENTITY)') IS NULL
    CREATE USER [$(APP_IDENTITY)] FROM EXTERNAL PROVIDER;
IF DATABASE_PRINCIPAL_ID(N'$(QUERY_IDENTITY)') IS NULL
    CREATE USER [$(QUERY_IDENTITY)] FROM EXTERNAL PROVIDER;
GO

IF DATABASE_PRINCIPAL_ID(N'advisor_app') IS NULL CREATE ROLE advisor_app;
IF DATABASE_PRINCIPAL_ID(N'advisor_query') IS NULL CREATE ROLE advisor_query;
GO

GRANT SELECT ON SCHEMA::dbo TO advisor_app;
GRANT INSERT, UPDATE, DELETE ON dbo.audit_events TO advisor_app;
GRANT INSERT, UPDATE ON dbo.conversations TO advisor_app;
GRANT INSERT ON dbo.conversation_messages TO advisor_app;
GRANT INSERT ON dbo.recommendations TO advisor_app;
GRANT INSERT ON dbo.forecast_results TO advisor_app;
GRANT INSERT, DELETE ON dbo.anomalies TO advisor_app;
GRANT INSERT, DELETE ON dbo.documents TO advisor_app;
DENY DELETE, UPDATE ON dbo.audit_events TO advisor_app;
GO

GRANT SELECT ON dbo.facilities TO advisor_query;
GRANT SELECT ON dbo.emissions TO advisor_query;
GRANT SELECT ON dbo.energy TO advisor_query;
GRANT SELECT ON dbo.production TO advisor_query;
GRANT SELECT ON dbo.sensors TO advisor_query;
GRANT SELECT ON dbo.sensor_readings TO advisor_query;
GRANT SELECT ON dbo.sustainability_metrics TO advisor_query;
GRANT SELECT ON dbo.anomalies TO advisor_query;
GO

ALTER ROLE advisor_app ADD MEMBER [$(APP_IDENTITY)];
ALTER ROLE advisor_query ADD MEMBER [$(QUERY_IDENTITY)];
GO
