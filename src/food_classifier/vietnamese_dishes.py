"""Catalog of Vietnamese food classes proposed for crawling beyond Food-101."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import unicodedata


@dataclass(frozen=True)
class Dish:
    canonical_label: str
    vietnamese_name: str
    priority: str
    region: str
    english_queries: tuple[str, ...]
    vietnamese_queries: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    @property
    def queries(self) -> tuple[str, ...]:
        return self.vietnamese_queries + self.english_queries


def _ascii_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


_RAW: tuple[tuple[str, str, str, str, str], ...] = (
    ("banh_mi", "Bánh mì", "P0", "nationwide", "Vietnamese baguette sandwich"),
    ("bun_bo_hue", "Bún bò Huế", "P0", "Hue", "Hue beef noodle soup"),
    ("bun_cha", "Bún chả", "P0", "Hanoi", "Hanoi grilled pork noodles"),
    ("bun_rieu", "Bún riêu", "P0", "nationwide", "Vietnamese crab noodle soup"),
    ("bun_thit_nuong", "Bún thịt nướng", "P0", "South Vietnam", "Vietnamese grilled pork vermicelli"),
    ("bun_dau_mam_tom", "Bún đậu mắm tôm", "P0", "North Vietnam", "Vietnamese tofu vermicelli platter"),
    ("bun_mam", "Bún mắm", "P0", "Mekong Delta", "Vietnamese fermented fish noodle soup"),
    ("bun_oc", "Bún ốc", "P0", "Hanoi", "Vietnamese snail noodle soup"),
    ("bun_moc", "Bún mọc", "P0", "North Vietnam", "Vietnamese pork meatball noodles"),
    ("mi_quang", "Mì Quảng", "P0", "Quang Nam", "Quang-style turmeric noodles"),
    ("cao_lau", "Cao lầu", "P0", "Hoi An", "Hoi An pork noodles"),
    ("hu_tieu", "Hủ tiếu", "P0", "South Vietnam", "Vietnamese clear pork noodle soup"),
    ("banh_canh", "Bánh canh", "P0", "nationwide", "Vietnamese thick tapioca noodles"),
    ("banh_da_cua", "Bánh đa cua", "P0", "Hai Phong", "Hai Phong crab red noodles"),
    ("com_tam", "Cơm tấm", "P0", "South Vietnam", "Vietnamese broken rice pork"),
    ("com_ga_hoi_an", "Cơm gà Hội An", "P0", "Hoi An", "Vietnamese Hoi An chicken rice"),
    ("com_hen", "Cơm hến", "P0", "Hue", "Hue clam rice"),
    ("com_nieu", "Cơm niêu", "P0", "nationwide", "Vietnamese clay pot rice"),
    ("bo_kho", "Bò kho", "P0", "nationwide", "Vietnamese beef stew"),
    ("bo_luc_lac", "Bò lúc lắc", "P0", "South Vietnam", "Vietnamese shaking beef"),
    ("bo_nhung_dam", "Bò nhúng dấm", "P0", "South Vietnam", "Vietnamese vinegar hot pot beef"),
    ("thit_kho_trung", "Thịt kho trứng", "P0", "South Vietnam", "Vietnamese caramelized pork eggs"),
    ("ca_kho_to", "Cá kho tộ", "P0", "South Vietnam", "Vietnamese caramelized clay pot fish"),
    ("cha_ca_la_vong", "Chả cá Lã Vọng", "P0", "Hanoi", "Hanoi turmeric fish"),
    ("nem_nuong", "Nem nướng", "P0", "Central Vietnam", "Vietnamese grilled pork sausage"),
    ("nem_lui", "Nem lụi", "P0", "Hue", "Hue lemongrass pork skewers"),
    ("chao_tom", "Chạo tôm", "P0", "Hue", "Vietnamese sugarcane shrimp"),
    ("lau_mam", "Lẩu mắm", "P0", "Mekong Delta", "Vietnamese fermented fish hot pot"),
    ("lau_ga_la_e", "Lẩu gà lá é", "P0", "Da Lat", "Da Lat chicken hot pot"),
    ("lau_de", "Lẩu dê", "P0", "nationwide", "Vietnamese goat hot pot"),
    ("banh_xeo", "Bánh xèo", "P1", "nationwide", "Vietnamese sizzling pancake"),
    ("banh_khot", "Bánh khọt", "P1", "Vung Tau", "Vietnamese crispy shrimp pancakes"),
    ("banh_beo", "Bánh bèo", "P1", "Hue", "Hue steamed rice cakes"),
    ("banh_cuon", "Bánh cuốn", "P1", "North Vietnam", "Vietnamese steamed rice rolls"),
    ("banh_uot", "Bánh ướt", "P1", "nationwide", "Vietnamese fresh rice sheets"),
    ("banh_bot_loc", "Bánh bột lọc", "P1", "Hue", "Vietnamese tapioca shrimp dumplings"),
    ("banh_can", "Bánh căn", "P1", "Nha Trang", "Vietnamese mini rice pancakes"),
    ("banh_it_tran", "Bánh ít trần", "P1", "Central Vietnam", "Vietnamese sticky rice dumplings"),
    ("banh_gio", "Bánh giò", "P1", "North Vietnam", "Vietnamese pyramid rice dumpling"),
    ("banh_chung", "Bánh chưng", "P1", "North Vietnam", "Vietnamese square sticky rice cake"),
    ("banh_tet", "Bánh tét", "P1", "South Vietnam", "Vietnamese cylindrical sticky rice cake"),
    ("banh_dap", "Bánh đập", "P1", "Hoi An", "Hoi An rice cracker"),
    ("goi_ga", "Gỏi gà", "P1", "nationwide", "Vietnamese chicken salad"),
    ("goi_du_du", "Gỏi đu đủ", "P1", "South Vietnam", "Vietnamese green papaya salad"),
    ("goi_ca_trich", "Gỏi cá trích", "P1", "Phu Quoc", "Phu Quoc herring salad"),
    ("goi_sen", "Gỏi ngó sen", "P1", "South Vietnam", "Vietnamese lotus stem salad"),
    ("xoi_man", "Xôi mặn", "P1", "South Vietnam", "Vietnamese savory sticky rice"),
    ("xoi_xeo", "Xôi xéo", "P1", "Hanoi", "Hanoi mung bean sticky rice"),
    ("xoi_ngo", "Xôi ngô", "P1", "North Vietnam", "Vietnamese corn sticky rice"),
    ("banh_hoi", "Bánh hỏi", "P1", "Central and South Vietnam", "Vietnamese woven rice vermicelli"),
    ("canh_chua_ca", "Canh chua cá", "P2", "South Vietnam", "Vietnamese sour fish soup"),
    ("canh_kho_qua_nhoi_thit", "Canh khổ qua nhồi thịt", "P2", "South Vietnam", "Vietnamese stuffed bitter melon soup"),
    ("canh_bong", "Canh bóng", "P2", "North Vietnam", "Vietnamese pork rind soup"),
    ("ca_loc_nuong_trui", "Cá lóc nướng trui", "P2", "Mekong Delta", "Mekong grilled snakehead fish"),
    ("ca_ba_sa_kho", "Cá basa kho", "P2", "Mekong Delta", "Vietnamese braised basa fish"),
    ("tom_rim", "Tôm rim", "P2", "nationwide", "Vietnamese caramelized shrimp"),
    ("thit_luong", "Thịt luộc", "P2", "nationwide", "Vietnamese boiled pork"),
    ("thit_heo_quay", "Thịt heo quay", "P2", "nationwide", "Vietnamese crispy roast pork"),
    ("ga_nuong_muoi_ot", "Gà nướng muối ớt", "P2", "South Vietnam", "Vietnamese chili salt grilled chicken"),
    ("ga_nuong_com_lam", "Gà nướng cơm lam", "P2", "Central Highlands", "Vietnamese grilled chicken bamboo rice"),
    ("muoi_ot_xanh", "Muối ớt xanh", "P2", "Central Vietnam", "Vietnamese green chili salt sauce"),
    ("mam_tom", "Mắm tôm", "P2", "North Vietnam", "Vietnamese shrimp paste"),
    ("muc_nuong", "Mực nướng", "P2", "nationwide", "Vietnamese grilled squid"),
    ("oc_len_xao_dua", "Ốc len xào dừa", "P2", "South Vietnam", "Vietnamese coconut curry snails"),
    ("ngheu_hap_sa", "Nghêu hấp sả", "P2", "nationwide", "Vietnamese lemongrass steamed clams"),
    ("che_ba_mau", "Chè ba màu", "P3", "nationwide", "Vietnamese three color dessert"),
    ("che_troi_nuoc", "Chè trôi nước", "P3", "South Vietnam", "Vietnamese glutinous rice balls dessert"),
    ("che_dau_den", "Chè đậu đen", "P3", "nationwide", "Vietnamese black bean sweet soup"),
    ("che_chuoi", "Chè chuối", "P3", "South Vietnam", "Vietnamese banana coconut dessert"),
    ("che_khoai", "Chè khoai", "P3", "nationwide", "Vietnamese sweet potato dessert"),
    ("chuoi_nep_nuong", "Chuối nếp nướng", "P3", "South Vietnam", "Vietnamese grilled banana sticky rice"),
    ("sua_chua_nep_cam", "Sữa chua nếp cẩm", "P3", "North Vietnam", "Vietnamese yogurt black rice"),
    ("banh_fl_nuong", "Bánh flan nướng", "P3", "nationwide", "Vietnamese baked flan"),
    ("banh_bo", "Bánh bò", "P3", "South Vietnam", "Vietnamese honeycomb cake"),
    ("banh_da_lon", "Bánh da lợn", "P3", "South Vietnam", "Vietnamese pandan layer cake"),
    ("banh_phu_the", "Bánh phu thê", "P3", "North Vietnam", "Vietnamese husband and wife cake"),
    ("banh_com", "Bánh cốm", "P3", "Hanoi", "Hanoi green rice cake"),
    ("ca_phe_sua_da", "Cà phê sữa đá", "P3", "nationwide", "Vietnamese condensed milk coffee"),
    ("nuoc_mia", "Nước mía", "P3", "nationwide", "Vietnamese sugarcane juice"),
    ("tra_da_via_he", "Trà đá vỉa hè", "P3", "nationwide", "Vietnamese iced tea"),
)


def _make_dishes() -> tuple[Dish, ...]:
    return tuple(
        Dish(label, name, priority, region, (english,), (name, f"{label.replace('_', ' ')} Vietnamese"))
        for label, name, priority, region, english in _RAW
    )


DISHES = _make_dishes()
EXCLUDED_OR_REVIEW = (
    ("pho", "already present in Food-101", "pho"),
    ("goi_cuon", "semantic overlap", "spring_rolls"),
    ("cha_gio", "semantic overlap", "spring_rolls"),
    ("com_chien", "semantic overlap", "fried_rice"),
    ("ca_ri_ga", "semantic overlap", "chicken_curry"),
    ("canh_chua", "semantic overlap", "hot_and_sour_soup"),
    ("banh_bao", "semantic overlap", "dumplings"),
    ("ha_cao", "semantic overlap", "dumplings"),
    ("xiu_mai", "semantic overlap", "dumplings"),
    ("banh_pancake", "review visual overlap", "pancakes"),
    ("nem_ran", "use cha_gio if creating a separate class", "spring_rolls"),
)


def export_catalog(path: Path) -> None:
    priorities = {priority: [dish.canonical_label for dish in DISHES if dish.priority == priority] for priority in ("P0", "P1", "P2", "P3")}
    payload = {
        "version": 1,
        "dataset": "vietnamese_food_candidates",
        "priorities": priorities,
        "classes": [asdict(dish) | {"queries": list(dish.queries)} for dish in DISHES],
        "excluded_or_review": [
            {"label": label, "reason": reason, "food101_class": food101_class}
            for label, reason, food101_class in EXCLUDED_OR_REVIEW
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
