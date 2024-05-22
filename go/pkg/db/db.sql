set time zone 'UTC';

-- requestLog

CREATE TABLE public.models (
    id int generated always as identity primary key,
    train_id text not null,
    name text not null,
    param jsonb not null,
    content bytea,
    created_at timestamp without time zone default now() not null
);

CREATE TABLE public.gatingresult (
    id int generated always as identity primary key,
    train_id text not null,
    p1 text not null,
    p2 text not null,
    score float not null,
    count int not null,
    created_at timestamp without time zone default now() not null
);

CREATE TABLE public.dataset (
    id int generated always as identity primary key,
    train_id text not null,
    c bytea,
    v bytea,
    p bytea,
    created_at timestamp without time zone default now() not null
);

