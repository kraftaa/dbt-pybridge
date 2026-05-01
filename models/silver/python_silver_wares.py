def model(dbt, session):
    dbt.config(
        materialized="table",
        indexes=[
            {"columns": ["id"]},
            {"columns": ["primary_discipline_id"]},
            {"columns": ["primary_category_id"]},
            {"columns": ["primary_subcategory_id"]},
        ],
    )

    dynamic_forms = dbt.ref("bronze_dynamic_forms")[["id", "name", "slug"]].rename(
        columns={
            "id": "dynamic_form_id",
            "name": "dynamic_form_name",
            "slug": "dynamic_form_slug",
        }
    )

    wares = dbt.ref("bronze_wares")
    beacons = dbt.ref("bronze_beacons")[["id", "name"]]

    discipline = beacons.rename(
        columns={
            "id": "primary_discipline_id",
            "name": "primary_discipline_name",
        }
    )
    category = beacons.rename(
        columns={
            "id": "primary_category_id",
            "name": "primary_category_name",
        }
    )
    subcategory = beacons.rename(
        columns={
            "id": "primary_subcategory_id",
            "name": "primary_subcategory_name",
        }
    )

    result = wares.merge(discipline, on="primary_discipline_id", how="left")
    result = result.merge(category, on="primary_category_id", how="left")
    result = result.merge(subcategory, on="primary_subcategory_id", how="left")
    result = result.merge(dynamic_forms, on="dynamic_form_id", how="left")

    return result[
        [
            "id",
            "uuid",
            "name",
            "slug",
            "description",
            "dynamic_form_id",
            "dynamic_form_name",
            "dynamic_form_slug",
            "primary_discipline_id",
            "primary_discipline_name",
            "primary_category_id",
            "primary_category_name",
            "primary_subcategory_id",
            "primary_subcategory_name",
            "created_at",
            "updated_at",
        ]
    ]
